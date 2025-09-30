import os, re, json, struct
import numpy as np
from src.utils.common import vprint, be_hex_float, sanitize_scale

def parse_trt_calib_cache_to_scales(cache_path: str, verbose: bool = False):
    """
    解析 TensorRT calibration cache，提取原始浮点 raw_f，并打印两种解释：
      (A) 作为 scale 使用: scale_as_is = raw_f
      (B) 作为 amax 使用:  scale_from_amax = raw_f / 127.0   # symmetric int8
    函数返回值保持不变：沿用 (B) 的结果 {name: raw_f/127.0}，仅新增 verbose 观测信息。
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"calibration cache not found: {cache_path}")

    with open(cache_path, "rb") as f:
        raw = f.read()
    try:
        txt = raw.decode("utf-8", errors="ignore")
    except Exception:
        txt = raw.decode("latin-1", errors="ignore")

    pat = re.compile(r"^([^:]+):\s*([0-9a-fA-F]{8})\s*$")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

    scales = {}
    raw_map = {}
    header_printed = False
    skipped = 0

    for ln in lines:
        if ln.startswith("TRT-"):
            if not header_printed:
                vprint(verbose, f"[Cache] Header: {ln}")
                header_printed = True
            continue
        m = pat.match(ln)
        if not m:
            continue
        name, hex8 = m.group(1).strip(), m.group(2).lower()
        try:
            raw_f = struct.unpack(">f", bytes.fromhex(hex8))[0]
        except Exception:
            skipped += 1
            continue
        if not np.isfinite(raw_f) or raw_f <= 0.0:
            skipped += 1
            continue

        raw_map[name] = float(raw_f)
        scales[name] = float(raw_f) / 127.0  # 【保持原行为】：当作 amax，再除以 127 得 scale

    vprint(verbose, f"[Cache] parsed entries: {len(scales)}, skipped={skipped}")
    if verbose and raw_map:
        vprint(verbose, "[Cache] preview (first 10, raw & two interpretations):")
        for i, (k, raw_f) in enumerate(list(raw_map.items())[:10]):
            scale_as_is = raw_f
            scale_from_amax = raw_f / 127.0
            vprint(verbose, f"  {i:02d}. {k}: raw_f={raw_f:.6g}  "
                            f"(as_scale={scale_as_is:.6g}, from_amax/127={scale_from_amax:.6g})")

    return scales


def export_txt_json_from_profile(network, header, txt_path, json_path, ranges_dict=None):
    """
    从 network 的层输出张量名导出：
      txt: header + "name: hex_be"
      json: { name: {scale: float, min:0, max:0, offset:0}, ... }
    注意：TRT Python API 不直接给出每层 dynamic range；此处以 1.0 作为保底，
    你若用 trtexec --exportProfile 或其他方式拿到了真实 amax，可填到 ranges_dict。
    """
    items = []
    jroot = {}
    # 先网络输入
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if inp and inp.name:
            sc = 1.0
            if ranges_dict and inp.name in ranges_dict:
                sc = sanitize_scale(ranges_dict[inp.name], fallback=1.0)
            items.append((inp.name, sc))

    # 再各层输出
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for oi in range(layer.num_outputs):
            t = layer.get_output(oi)
            if t is None or not t.name:
                continue
            sc = 1.0
            if ranges_dict and t.name in ranges_dict:
                sc = sanitize_scale(ranges_dict[t.name], fallback=1.0)
            items.append((t.name, sc))

    with open(txt_path, "w") as f:
        f.write(header + "\n")
        for name, sc in items:
            f.write(f"{name}: {be_hex_float(sc)}\n")
    print(f"[OK] wrote calib.txt : {txt_path}  (tensors={len(items)})")

    for name, sc in items:
        jroot[name] = {"scale": float(sc), "min": 0, "max": 0, "offset": 0}
    with open(json_path, "w") as f:
        json.dump(jroot, f, indent=2)
    print(f"[OK] wrote calib.json: {json_path}")
