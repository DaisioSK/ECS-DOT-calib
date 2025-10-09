# =========================================================
# File: nvdla_vp/scripts/gen_caffemodel.py
# Purpose: Generate random-initialized Caffe model weights
# Author: Seakon Liu
# =========================================================
from __future__ import print_function
import os, sys, numpy as np
import caffe

# ---- Basic paths (relative to workspace root) ----
PROTOTXT = "nvdla_vp/models/cifar_simplecnn.prototxt"
OUTPATH  = "nvdla_vp/models/cifar_simplecnn.caffemodel"

def main():
    caffe.set_mode_cpu()
    print("[INFO] Current working dir:", os.getcwd())
    print("[INFO] Loading prototxt:", PROTOTXT)
    assert os.path.exists(PROTOTXT), "Prototxt not found: %s" % PROTOTXT

    net = caffe.Net(PROTOTXT, caffe.TEST)

    has_param = False
    for lname, blobs in net.params.items():
        has_param = True
        shapes = []
        for i, blob in enumerate(blobs):
            shape = blob.data.shape
            shapes.append(shape)
            if i == 0:  # weights
                blob.data[...] = np.random.randn(*shape).astype(np.float32) * 0.01
            else:       # bias
                blob.data[...] = 0.0
        print("[INIT] %-16s params=%s" % (lname, shapes))

    if not has_param:
        print("[WARN] No parameterized layers (no Conv/FC). Saving anyway.")

    net.save(OUTPATH)
    print("[OK] Saved:", OUTPATH, "size:", os.path.getsize(OUTPATH), "bytes")

if __name__ == "__main__":
    main()
