import numpy as np
from sklearn.metrics import roc_auc_score

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

def auc_multiclass(probs, y_true):
    C = probs.shape[1]
    Y = np.eye(C, dtype=np.float32)[y_true]
    try:
        return {
            "macro": float(roc_auc_score(Y, probs, average="macro", multi_class="ovr")),
            "micro": float(roc_auc_score(Y, probs, average="micro", multi_class="ovr")),
        }
    except Exception:
        return None