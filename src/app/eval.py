import json, time
import numpy as np
from pathlib import Path
from src.core.trt_runner import TrtRunner
from src.utils.evaluation import softmax, topk_acc, confusion_matrix, precision_recall_f1_from_cm, auc_multiclass

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
