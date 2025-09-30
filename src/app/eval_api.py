# src/app/eval_api.py
import os, json, time
import numpy as np
from pathlib import Path
from src.core.trt_runner import TrtRunner
# 直接复制你 eval.py 里的这些函数过来或就地粘贴：
#   softmax, topk_acc, confusion_matrix, precision_recall_f1_from_cm, try_auc_multiclass

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

def run_eval_on_npz(engine: str, npz: str, batch_size: int, outdir: str):
    outdir_p = Path(outdir); outdir_p.mkdir(parents=True, exist_ok=True)
    d = np.load(npz)
    X = d["imgs"].astype(np.float32, copy=False)
    y = d["labels"].astype(np.int64,  copy=False)
    N, C, H, W = X.shape
    runner = TrtRunner(engine)

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
    auc = try_auc_multiclass(probs, y)

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
