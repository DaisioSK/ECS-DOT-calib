#!/usr/bin/env python
# -*- coding: utf-8 -*-

# compatible to python2 

# =========================================================
# File: nvdla_vp/scripts/gen_caffemodel.py
# Purpose: Generate random-initialized Caffe model weights
# Author: Seakon Liu
# =========================================================

from __future__ import print_function
import os, sys, argparse, numpy as np
import caffe

def gen_weights(proto_path, out_path):
    caffe.set_mode_cpu()
    if not os.path.exists(proto_path):
        raise RuntimeError("Prototxt not found: {}".format(proto_path))
    net = caffe.Net(proto_path, caffe.TEST)

    # 随机初始化所有带参数层
    for lname, blobs in net.params.items():
        W = blobs[0].data
        blobs[0].data[...] = np.random.randn(*W.shape).astype(np.float32) * 0.01
        if len(blobs) > 1:
            blobs[1].data[...] = 0.0
        print("[INIT] {} {}".format(lname, [b.data.shape for b in blobs]))

    # —— 自动决策 3→4 补零（默认不做；仅当网络本身 in_c=4 且现权重=3 时才补）——
    # 通常 Caffe 的权重 shape == 网络定义 shape，不会出现 in_c 不匹配的情况。
    # 此逻辑只是兜底：如果首层需要 4 通道但权重是 3 通道，就补零到 4。
    try:
        first = list(net.params.keys())[0]
        W = net.params[first][0].data
        if len(W.shape) == 4:
            oc, ic, kh, kw = W.shape
            # 仅当“网络需要 4 通道而当前是 3 通道”时补零
            if ic == 3 and _net_input_channels(net) == 4:
                W4 = np.zeros((oc, 4, kh, kw), dtype=W.dtype)
                W4[:, :3, :, :] = W
                net.params[first][0].data[...] = W4
                print("[PAD ] {}: in_c 3 -> 4 (zeros) to satisfy network expecting 4ch".format(first))
            else:
                print("[INFO] No pad: first conv in_c={}, net_in_c={}".format(ic, _net_input_channels(net)))
    except Exception as e:
        print("[WARN] Auto pad check skipped: {}".format(e))

    d = os.path.dirname(out_path)
    if d and not os.path.exists(d):
        os.makedirs(d)
    net.save(out_path)
    print("[OK ] Saved: {}".format(out_path))

def _net_input_channels(net):
    # 尝试从 input blob 维度推断 NCHW 的 C；失败则返回 3
    try:
        blob = net.blobs.get('data', None)
        if blob is not None and len(blob.data.shape) >= 2:
            return int(blob.data.shape[1])
    except Exception:
        pass
    return 3

def main():
    ap = argparse.ArgumentParser("prototxt -> random caffemodel (auto policy)")
    ap.add_argument("prototxt_in")
    ap.add_argument("caffemodel_out")
    args = ap.parse_args()
    gen_weights(args.prototxt_in, args.caffemodel_out)

if __name__ == "__main__":
    main()
