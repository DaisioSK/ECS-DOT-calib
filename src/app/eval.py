import json, time
import numpy as np
from pathlib import Path
from src.core.trt_runner import TrtRunner
from src.utils.evaluation import softmax, topk_acc, confusion_matrix, precision_recall_f1_from_cm, auc_multiclass

def run_eval(data_format, engine, args={}):
    if data_format == "npz":
        return run_eval_on_npz(
            engine=engine, 
            npz=args["npz"],
            batch_size=args["batch_size"],
            outdir=args["outdir"],
        )
     

def run_eval_on_npz(engine: str, npz: str, batch_size: int, outdir: str):
    outdir_p = Path(outdir); outdir_p.mkdir(parents=True, exist_ok=True)
    z = np.load(npz, allow_pickle=False)

    # 1) 兼容多种键名
    img_key = "imgs" if "imgs" in z.files else ("images" if "images" in z.files else None)
    if img_key is None:
        raise KeyError(f"No images array found in {npz}; tried keys: imgs, images. Found: {z.files}")

    lbl_key = "labels" if "labels" in z.files else ("y" if "y" in z.files else None)
    if lbl_key is None:
        # 评测必须有标签；若没有就报错（或退化为只跑推理）
        raise KeyError(f"No labels array found in {npz}; tried keys: labels, y. Found: {z.files}")

    X = z[img_key].astype(np.float32, copy=False)  # 期望 NCHW
    y = z[lbl_key].astype(np.int64, copy=False)

    # 2) 若意外是 NHWC，自动转成 NCHW（稳妥起见）
    if X.ndim == 4 and X.shape[1] not in (1,3):   # 可能是 NHWC
        X = np.transpose(X, (0,3,1,2)).copy()

    N, C, H, W = X.shape
    runner = TrtRunner(engine)

    # 选 batch 大小
    if runner.is_implicit:
        cap = runner.max_batch_size or 1
        bs = min(max(1, int(batch_size)), cap)
        if batch_size > bs:
            print(f"[Note] Implicit engine cap batch to {bs} (requested {batch_size})")
    else:
        bs = runner.fixed_batch if runner.fixed_batch is not None else max(1, int(batch_size))
        if runner.fixed_batch is not None and batch_size != bs:
            print(f"[Note] Engine fixed batch={bs}; override --bs {batch_size} -> {bs}")

    all_logits = []
    t0 = time.time()
    for i in range(0, N, bs):
        logits = runner.infer(X[i:i+bs])
        if logits.ndim > 2: logits = logits.reshape(logits.shape[0], -1)
        all_logits.append(logits)
    t1 = time.time()

    logits = np.concatenate(all_logits, axis=0)
    num_classes = logits.shape[1]
    probs = softmax(logits)
    pred  = probs.argmax(1)

    top1 = float((pred == y).mean()); acc = top1
    top5 = topk_acc(logits, y, k=5)
    cm = confusion_matrix(pred, y, num_classes)
    per_cls, macro, micro, weighted = precision_recall_f1_from_cm(cm)
    auc = auc_multiclass(probs, y)

    dur = t1 - t0
    throughput = N / dur if dur > 0 else float("nan")
    batches = int(np.ceil(N / bs))
    per_batch_ms = (dur * 1000.0) / max(1, batches)

    report = {
        "engine": str(Path(engine).resolve()),
        "npz": str(Path(npz).resolve()),
        "num_images": int(N),
        "batch_size": int(bs),
        "accuracy": acc,
        "top1": top1,
        "top5": None if np.isnan(top5) else float(top5),
        "macro": macro, "micro": micro, "weighted": weighted,
        "auc_ovr": auc,
        "throughput_img_s": throughput,
        "per_batch_latency_ms": per_batch_ms,
        # dtype 信息可按需补充
    }
    rep_json = outdir_p / "report.json"
    with open(rep_json, "w") as f:
        json.dump(report, f, indent=2)
    cm_csv = outdir_p / "confusion_matrix.csv"
    np.savetxt(cm_csv, cm, fmt="%d", delimiter=",")
    print(f"[Saved] {rep_json}")
    print(f"[Saved] {cm_csv}")
    return {"report_json": str(rep_json), "cm_csv": str(cm_csv)}
