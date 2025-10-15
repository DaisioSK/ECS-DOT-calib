#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX -> Caffe prototxt (Lean; optional FEATURE path)
- 默认不插入 Power/Scale，第一层可直接 Conv
- 若传入 --force-feature，则插入：
    1) Power(feat0): identity
    2) Scale(feat_scale): identity -> feat1
  并让第一层真实算子从 feat1 开始（NVDLA 可走 FEATURE 路径）
- 层名仅保留 [0-9a-zA-Z_]
- 覆盖 op: Conv / Relu / (Max|Average)Pool / Flatten / Gemm|MatMul / Softmax
- 不额外添加 Softmax：是否存在完全取决于 ONNX
"""
import os, argparse, re
import onnx

_SAN = re.compile(r'[^0-9a-zA-Z_]+')

def _san_name(s: str) -> str:
    s = s or "noname"
    s = _SAN.sub("_", s)
    return s if s else "noname"

def _shape(vi):
    """Return (N,C,H,W) from ONNX ValueInfo; unknown dims -> 1; fallback to (1,3,32,32)."""
    shp = []
    tt = vi.type.tensor_type
    if tt and tt.shape:
        for d in tt.shape.dim:
            v = getattr(d, "dim_value", 0) or 0
            shp.append(int(v) if v > 0 else 1)
    if len(shp) < 4:
        shp = [1, 3, 32, 32]
    return tuple(shp[:4])

def onnx_to_spec(onnx_path: str):
    m = onnx.load(onnx_path)
    g = m.graph
    n, c, h, w = _shape(g.input[0]) if g.input else (1, 3, 32, 32)
    layers = []
    for node in g.node:
        t = node.op_type
        nname = node.name or (node.output[0] if node.output else "noname")
        nname = _san_name(nname)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

        if t == "Conv":
            k = (attrs.get("kernel_shape", [3, 3]) or [3, 3])[0]
            s = (attrs.get("strides", [1, 1]) or [1, 1])[0]
            p = (attrs.get("pads", [0, 0, 0, 0]) or [0, 0, 0, 0])[0]
            layers.append({"type": "conv", "name": nname, "out": 64, "k": int(k), "s": int(s), "p": int(p), "bias": True})

        elif t == "Relu":
            if layers and layers[-1]["type"] in ("conv", "fc"):
                layers[-1]["relu"] = True

        elif t in ("MaxPool", "AveragePool"):
            k = (attrs.get("kernel_shape", [2, 2]) or [2, 2])[0]
            s = (attrs.get("strides", [2, 2]) or [2, 2])[0]
            p = (attrs.get("pads", [0, 0, 0, 0]) or [0, 0, 0, 0])[0]
            mode = "max" if t == "MaxPool" else "ave"
            layers.append({"type": "pool", "name": nname, "mode": mode, "k": int(k), "s": int(s), "p": int(p)})

        elif t == "Flatten":
            layers.append({"type": "flatten", "name": nname})

        elif t in ("Gemm", "MatMul"):
            layers.append({"type": "fc", "name": nname, "out": 256, "bias": True})

        elif t == "Softmax":
            layers.append({"type": "softmax", "name": "prob"})

        # 其他算子可按需扩展

    name = _san_name(os.path.splitext(os.path.basename(onnx_path))[0])
    return {"name": name, "input": (int(n), int(c), int(h), int(w)), "layers": layers}

def emit_prototxt(spec, out_path, force_feature=False):
    n, c, h, w = spec["input"]
    L = spec["layers"]

    lines = []
    lines += [
        f'name: "{spec["name"]}"',
        'input: "data"',
        f'input_shape {{ dim: {n} dim: {c} dim: {h} dim: {w} }}',
        "",
    ]

    # 是否强制 FEATURE 路径
    top = "data"
    if force_feature:
        lines += [
            'layer { name: "feat0" type: "Power" bottom: "data" top: "feat0"',
            '  power_param { power: 1.0 scale: 1.0 shift: 0.0 } }',
            ''
        ]
        lines += [
            'layer { name: "feat_scale" type: "Scale" bottom: "feat0" top: "feat1"',
            '  scale_param {',
            '    bias_term: false',
            '    filler { type: "constant" value: 1 }',
            '  }',
            '}',
            ''
        ]
        top = "feat1"

    # 逐层发射
    for l in L:
        t = l["type"]
        if t == "conv":
            nm = _san_name(l["name"])
            lines += [
                f'layer {{ name: "{nm}" type: "Convolution" bottom: "{top}" top: "{nm}"',
                f'  convolution_param {{ num_output: {l["out"]} kernel_size: {l["k"]} stride: {l["s"]} pad: {l["p"]}',
                ('    bias_term: true\n    weight_filler { type: "xavier" }\n    bias_filler { type: "constant" }'
                 if l.get("bias", True) else
                 '    bias_term: false\n    weight_filler { type: "xavier" }'),
                '  } }'
            ]
            if l.get("relu"):
                lines += [f'layer {{ name: "relu_{nm}" type: "ReLU" bottom: "{nm}" top: "{nm}" }}']
            top = nm

        elif t == "pool":
            nm = _san_name(l["name"])
            pool_kw = "MAX" if l.get("mode", "max").lower() == "max" else "AVE"
            lines += [
                f'layer {{ name: "{nm}" type: "Pooling" bottom: "{top}" top: "{nm}"',
                f'  pooling_param {{ pool: {pool_kw} kernel_size: {l["k"]} stride: {l["s"]} pad: {l["p"]} }} }}'
            ]
            top = nm

        elif t == "flatten":
            nm = _san_name(l.get("name", "flatten"))
            lines += [f'layer {{ name: "{nm}" type: "Flatten" bottom: "{top}" top: "flat" }}']
            top = "flat"

        elif t == "fc":
            nm = _san_name(l["name"])
            lines += [
                f'layer {{ name: "{nm}" type: "InnerProduct" bottom: "{top}" top: "{nm}"',
                f'  inner_product_param {{ num_output: {l["out"]}',
                ('    bias_term: true\n    weight_filler { type: "xavier" }\n    bias_filler { type: "constant" }'
                 if l.get("bias", True) else
                 '    bias_term: false\n    weight_filler { type: "xavier" }'),
                '  } }'
            ]
            if l.get("relu"):
                lines += [f'layer {{ name: "relu_{nm}" type: "ReLU" bottom: "{nm}" top: "{nm}" }}']
            top = nm

        elif t == "softmax":
            nm = _san_name(l.get("name", "prob"))
            lines += [f'layer {{ name: "{nm}" type: "Softmax" bottom: "{top}" top: "{nm}" }}']
            top = nm

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[OK] prototxt ->", out_path)

def main():
    ap = argparse.ArgumentParser("ONNX -> Caffe prototxt (lean; optional FEATURE path)")
    ap.add_argument("onnx_path")
    ap.add_argument("--out", required=True, help="output prototxt path")
    ap.add_argument("--force-feature", action="store_true",
                    help="Force FEATURE path by inserting identity Power+Scale before the first real layer")
    args = ap.parse_args()
    spec = onnx_to_spec(args.onnx_path)
    emit_prototxt(spec, args.out, force_feature=args.force_feature)

if __name__ == "__main__":
    main()
