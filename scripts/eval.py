#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, time, os, json, warnings
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa

# ---------- metrics ----------
def softmax(x):
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def topk_acc(logits, labels, k):
    if logits.shape[1] < k: return float("nan")
    topk_idx = np.argpartition(-logits, k-1, axis=1)[:, :k]
    row = np.arange(logits.shape[0])[:, None]
    topk_sorted = topk_idx[row, np.argsort(-logits[row, topk_idx])]
    return float((topk_sorted[:, :k] == labels[:, None]).any(axis=1).mean())

def confusion_matrix(pred, labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, pred): cm[t, p] += 1
    return cm

def precision_recall_f1_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_c = np.where(tp+fp>0, tp/(tp+fp), 0.0)
        rec_c  = np.where(tp+fn>0, tp/(tp+fn), 0.0)
        f1_c   = np.where(prec_c+rec_c>0, 2*prec_c*rec_c/(prec_c+rec_c), 0.0)
    support = cm.sum(axis=1).astype(np.float64)
    total = support.sum()
    macro = {"precision": float(np.mean(prec_c)),
             "recall":    float(np.mean(rec_c)),
             "f1":        float(np.mean(f1_c))}
    w = np.where(support>0, support/np.maximum(total,1.0), 0.0)
    weighted = {"precision": float(np.sum(prec_c*w)),
                "recall":    float(np.sum(rec_c*w)),
                "f1":        float(np.sum(f1_c*w))}
    TP, FP, FN = float(tp.sum()), float(fp.sum()), float(fn.sum())
    micro_prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    micro_rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    micro_f1   = (2*micro_prec*micro_rec/(micro_prec+micro_rec)) if (micro_prec+micro_rec)>0 else 0.0
    micro = {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1}
    per_class = [{"precision": float(prec_c[i]),
                  "recall":    float(rec_c[i]),
                  "f1":        float(f1_c[i]),
                  "support":   int(support[i])} for i in range(len(tp))]
    return per_class, macro, micro, weighted

def try_auc_multiclass(probs, y_true):
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    C = probs.shape[1]
    Y = np.eye(C, dtype=np.float32)[y_true]
    try:
        return {
            "macro": float(roc_auc_score(Y, probs, average="macro", multi_class="ovr")),
            "micro": float(roc_auc_score(Y, probs, average="micro", multi_class="ovr")),
        }
    except Exception:
        return None

# ---------- TRT dtype (NumPy 2 safe) ----------
def safe_nptype(dtype: trt.DataType):
    if dtype == trt.DataType.FLOAT:   return np.float32
    if dtype == trt.DataType.HALF:    return np.float16
    if dtype == trt.DataType.INT8:    return np.int8
    if dtype == trt.DataType.INT32:   return np.int32
    if dtype == trt.DataType.BOOL:    return np.bool_
    if hasattr(trt.DataType, "UINT8") and dtype == trt.DataType.UINT8: return np.uint8
    return np.float32

# ---------- TensorRT runner (8.x/9.x 兼容；优先新接口，无警告) ----------
class TrtRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None: raise RuntimeError("Failed to deserialize engine")
        self.context = self.engine.create_execution_context()
        if self.context is None: raise RuntimeError("Failed to create execution context")

        self.has_new_tensor_api = all(
            hasattr(self.engine, name) for name in
            ("get_tensor_name","get_tensor_mode","get_tensor_shape","get_tensor_dtype")
        )
        self.has_enqueue_v3 = hasattr(self.context, "enqueue_v3")

        if self.has_new_tensor_api:
            # 新接口：不会触发 DeprecationWarning
            self.all_tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_names  = [n for n in self.all_tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in self.all_tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            assert len(self.input_names)==1 and len(self.output_names)==1
            self.in_name  = self.input_names[0]
            self.out_name = self.output_names[0]
            self.in_shape_engine = list(self.engine.get_tensor_shape(self.in_name))
            self.fixed_batch = self.in_shape_engine[0] if self.in_shape_engine and self.in_shape_engine[0] != -1 else None
            self.in_dtype  = safe_nptype(self.engine.get_tensor_dtype(self.in_name))
            self.out_dtype = safe_nptype(self.engine.get_tensor_dtype(self.out_name))
            print(f"[TRT] input={self.in_name} shape@engine={self.in_shape_engine} dtype={self.in_dtype}  "
                  f"output={self.out_name} dtype={self.out_dtype}")
        else:
            # 老接口：只在新接口不可用时兜底（避免 warnings）
            self.input_indices  = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
            self.output_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
            assert len(self.input_indices)==1 and len(self.output_indices)==1
            self.in_idx = self.input_indices[0]
            self.out_idx = self.output_indices[0]
            self.in_name = self.engine.get_binding_name(self.in_idx)
            self.out_name= self.engine.get_binding_name(self.out_idx)
            self.in_shape_engine = list(self.engine.get_binding_shape(self.in_idx))
            self.fixed_batch = self.in_shape_engine[0] if self.in_shape_engine and self.in_shape_engine[0] != -1 else None
            self.in_dtype  = safe_nptype(self.engine.get_binding_dtype(self.in_idx))
            self.out_dtype = safe_nptype(self.engine.get_binding_dtype(self.out_idx))
            print(f"[TRT] input={self.in_name} shape@engine={self.in_shape_engine} dtype={self.in_dtype}  "
                  f"output={self.out_name} dtype={self.out_dtype}")

    def _set_shape(self, B, C, H, W):
        """
        动态引擎：设置 (B,C,H,W)
        固定引擎：允许 B <= fixed_batch（B < fixed 时由 infer 内部做 padding），B > fixed 仍然报错
        """
        if self.has_new_tensor_api:
            if self.in_shape_engine[0] == -1:
                # 动态 batch
                self.context.set_input_shape(self.in_name, (B, C, H, W))
            else:
                fixed = self.in_shape_engine[0]
                if B > fixed:
                    raise ValueError(f"Engine fixed batch={fixed}, got larger B={B}")
                # B <= fixed：不改 context 形状，保持 fixed，后续在 infer 里做 pad
        else:
            if self.in_shape_engine[0] == -1:
                # 动态 batch（老 API）
                self.context.set_binding_shape(self.in_idx, (B, C, H, W))
            else:
                fixed = self.in_shape_engine[0]
                if B > fixed:
                    raise ValueError(f"Engine fixed batch={fixed}, got larger B={B}")
                # B <= fixed：不设置 binding 形状，保持 fixed，infer 里做 pad

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        支持两种模式：
          - 动态引擎：按 B 设置形状直接跑
          - 固定引擎：若 B < fixed_batch，自动把输入 pad 到 fixed_batch，再在输出切回前 B 条
        Padding 策略：重复该 batch 的最后一张（更稳妥，避免 out-of-distribution）
        """
        assert x.ndim == 4, f"expect NCHW, got {x.shape}"
        B, C, H, W = x.shape

        # 形状检查/设置（允许 B<=fixed；B>fixed 直接报错）
        self._set_shape(B, C, H, W)

        # ---- 准备输入（必要时 padding）----
        pad_used = False
        if self.fixed_batch is not None and B < self.fixed_batch:
            # 固定 batch，引擎预期 self.fixed_batch；我们补齐
            fixed = self.fixed_batch
            xb = x.astype(self.in_dtype, copy=False)
            x_pad = np.empty((fixed, C, H, W), dtype=self.in_dtype)
            x_pad[:B] = xb
            x_pad[B:] = xb[-1:]     # 重复最后一张
            x_contig = np.ascontiguousarray(x_pad)
            pad_used = True
            B_effective = fixed
        else:
            # 动态引擎，或固定引擎但 B==fixed
            x_contig = np.ascontiguousarray(x.astype(self.in_dtype, copy=False))
            B_effective = B

        # ---- 计算输出 shape 并分配显存 ----
        if self.has_new_tensor_api:
            out_shape_engine = tuple(self.context.get_tensor_shape(self.out_name))
        else:
            out_shape_engine = tuple(self.context.get_binding_shape(self.out_idx))
        # out_shape_engine[0] 可能是 fixed 或 B（动态）；按引擎回报分配
        out_bytes = int(np.prod(out_shape_engine)) * np.dtype(self.out_dtype).itemsize

        d_in = d_out = None
        try:
            d_in  = cuda.mem_alloc(x_contig.nbytes)
            d_out = cuda.mem_alloc(out_bytes)
            cuda.memcpy_htod(d_in, x_contig)

            if self.has_new_tensor_api and self.has_enqueue_v3:
                self.context.set_tensor_address(self.in_name,  int(d_in))
                self.context.set_tensor_address(self.out_name, int(d_out))
                if not self.context.enqueue_v3(stream_handle=0):
                    raise RuntimeError("enqueue_v3 failed")
            else:
                if not self.has_new_tensor_api:
                    bindings = [0]*self.engine.num_bindings
                    bindings[self.in_idx]  = int(d_in)
                    bindings[self.out_idx] = int(d_out)
                else:
                    in_idx  = next(i for i in range(self.engine.num_bindings)
                                   if self.engine.get_binding_name(i)==self.in_name)
                    out_idx = next(i for i in range(self.engine.num_bindings)
                                   if self.engine.get_binding_name(i)==self.out_name)
                    bindings = [0]*self.engine.num_bindings
                    bindings[in_idx]  = int(d_in)
                    bindings[out_idx] = int(d_out)
                if not self.context.execute_v2(bindings):
                    raise RuntimeError("execute_v2 failed")

            h_out = np.empty(out_shape_engine, dtype=self.out_dtype)
            cuda.memcpy_dtoh(h_out, d_out)

        finally:
            if d_in:  d_in.free()
            if d_out: d_out.free()

        # 统一转 float32，必要时切片回真实 B
        y = h_out.astype(np.float32, copy=False)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)

        if pad_used and y.shape[0] >= B:
            y = y[:B]  # 丢弃由 padding 产生的尾部输出
            # 可选提示：只在首次发生时打印
            # print(f"[Eval] Padded batch from {B} to {self.fixed_batch}, outputs sliced back to {B}")

        return y


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Evaluate a TensorRT engine on classification npz")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--outdir", default="./eval_trt_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.npz)
    X = d["imgs"].astype(np.float32, copy=False)
    y = d["labels"].astype(np.int64,  copy=False)
    N, C, H, W = X.shape
    print(f"[Data] imgs={X.shape} labels={y.shape}")

    runner = TrtRunner(args.engine)

    # batch size
    bs = runner.fixed_batch if runner.fixed_batch is not None else max(1, int(args.bs))
    if runner.fixed_batch is not None and args.bs != bs:
        print(f"[Note] Engine is fixed batch={bs}; overriding --bs {args.bs} -> {bs}")

    # infer
    all_logits = []
    t0 = time.time()
    for i in range(0, N, bs):
        xb = X[i:i+bs]
        logits = runner.infer(xb)
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        all_logits.append(logits)
    t1 = time.time()

    logits = np.concatenate(all_logits, axis=0)
    num_classes = logits.shape[1]
    probs = softmax(logits)
    pred  = probs.argmax(1)

    # metrics
    top1 = float((pred == y).mean())
    acc  = top1  # 显式 accuracy 字段
    top5 = topk_acc(logits, y, k=5)
    cm = confusion_matrix(pred, y, num_classes)
    per_cls, macro, micro, weighted = precision_recall_f1_from_cm(cm)
    auc = try_auc_multiclass(probs, y)

    dur = t1 - t0
    throughput = N / dur if dur > 0 else float("nan")
    batches = int(np.ceil(N / bs))
    per_batch_ms = (dur * 1000.0) / max(1, batches)

    print("\n========== TRT Summary ==========")
    print(f"engine: {os.path.abspath(args.engine)}")
    print(f"accuracy: {acc:.4f}  | top1: {top1:.4f} | top5: {top5 if not np.isnan(top5) else float('nan'):.4f}")
    print(f"macro  P/R/F1: {macro}")
    print(f"micro  P/R/F1: {micro}")
    print(f"weighted P/R/F1: {weighted}")
    print(f"AUC (OvR): {auc if auc is not None else 'N/A'}")
    print(f"throughput: {throughput:.2f} img/s | per-batch latency: {per_batch_ms:.2f} ms")
    print("=================================\n")

    report = {
        "engine": os.path.abspath(args.engine),
        "npz": os.path.abspath(args.npz),
        "num_images": int(N),
        "batch_size": int(bs),
        "accuracy": acc,
        "top1": top1,
        "top5": None if np.isnan(top5) else top5,
        "macro": macro, "micro": micro, "weighted": weighted,
        "auc_ovr": auc,
        "throughput_img_s": throughput,
        "per_batch_latency_ms": per_batch_ms,
        "input_dtype": str(runner.in_dtype), "output_dtype": str(runner.out_dtype),
    }
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.savetxt(os.path.join(args.outdir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    print(f"[Saved] {os.path.join(args.outdir, 'report.json')}")
    print(f"[Saved] {os.path.join(args.outdir, 'confusion_matrix.csv')}")
    print("Done.")

if __name__ == "__main__":
    # 安静点：屏蔽我们没有调用的 DeprecationWarning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()

