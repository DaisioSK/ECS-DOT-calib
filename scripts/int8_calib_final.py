#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust INT8 calibration & export (TensorRT 8.5+ friendly)

Outputs:
  - --engine <.engine>  (INT8)
  - --cache  <.cache>   (binary calibration cache)
  - --calib-txt <.txt>  (header + name: big-endian hex float)
  - --calib-json <.json> (NVDLA-friendly JSON)

Features:
  - MinMax / Entropy calibrators
  - Pinned host + stream + sync (stable get_batch)
  - --force: ignore existing cache, do fresh calibration
  - Optional --sanity N: compare INT8 vs FP32 logits similarity quickly
"""

import os, json, struct, argparse, re   # >>> CHANGED: add re for cache parsing
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa

# 兼容 TRT 对 np.bool 的老引用
if not hasattr(np, "bool"):
    np.bool = np.bool_  # noqa

# ----------------- Verbose helper (NEW) -----------------
def vprint(verbose: bool, *args, **kwargs):   # >>> NEW
    if verbose:
        print(*args, **kwargs)

# ----------------- Utils -----------------
def be_hex_float(f: float) -> str:
    import struct as _st
    return _st.pack(">f", float(f)).hex()

def sanitize_scale(x: float, fallback: float = 1.0):
    if not np.isfinite(x) or x <= 0.0:
        return float(fallback)
    return float(x)

def np_nbytes(shape, dtype=np.float32):
    return int(np.prod(shape)) * np.dtype(dtype).itemsize

def trt_dtype_to_np(dt):
    """
    Map TensorRT DataType to numpy dtype for upload alignment.
    Default to float32 if unknown.
    """
    try:
        import tensorrt as trt  # already imported at top
        mapping = {
            trt.DataType.FLOAT:  np.float32,
            trt.DataType.HALF:   np.float16,
            trt.DataType.INT8:   np.int8,
            trt.DataType.UINT8:  np.uint8,
            trt.DataType.BOOL:   np.bool_,
        }
        return mapping.get(dt, np.float32)
    except Exception:
        return np.float32

def nchw_channel_amax(x: np.ndarray):         # >>> NEW: for verbose inspection
    """返回全局amax，以及按通道amax（shape [C]），便于观察输入分布是否合理。"""
    amax_global = float(np.max(np.abs(x))) if x.size else 0.0
    ch = None
    if x.ndim == 4 and x.shape[1] > 0:
        ch = np.max(np.abs(x), axis=(0, 2, 3)).astype(np.float32, copy=False)
    return amax_global, ch

# ----------------- Calibrators -----------------
class _BaseNPZCalibrator:
    def __init__(self, npz_path, batch_size, cache_file,
                 max_calib_batches=None, force=False,
                 in_dtype_np=np.float32, verbose=False):
        """
        in_dtype_np: numpy dtype that matches TensorRT binding dtype (e.g., np.float16 if input is HALF)
        """
        self.cache_file = cache_file
        self.force = force
        self.batch_size = int(batch_size)
        self.in_dtype_np = np.dtype(in_dtype_np)
        self.verbose = bool(verbose)

        arr = np.load(npz_path)["imgs"]
        if arr.ndim != 4:
            raise ValueError(f"npz imgs must be [N,C,H,W], got {arr.shape}")
        # 数据集保持 float32 便于统计；上传前再转换为 in_dtype_np
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        total = arr.shape[0]
        usable = (total // self.batch_size) * self.batch_size
        if (max_calib_batches is not None) and (int(max_calib_batches) > 0):
            cap = self.batch_size * int(max_calib_batches)
            usable = min(usable, cap)
        arr = arr[:usable]
        if arr.shape[0] == 0:
            raise ValueError("No calibration samples.")
        self.data = np.ascontiguousarray(arr)
        self.idx = 0

        # 按“上传 dtype”分配 pinned + device
        upload_itemsize = self.in_dtype_np.itemsize
        nbytes_upload = int(np.prod((self.batch_size,) + tuple(self.data.shape[1:]))) * upload_itemsize
        self.h_page = cuda.pagelocked_empty(nbytes_upload, dtype=np.uint8)
        self.d_in = cuda.mem_alloc(nbytes_upload)
        self.stream = cuda.Stream()

        if self.verbose:
            print(f"[Calib] samples={self.data.shape[0]}  bs={self.batch_size}  "
                  f"batches={self.data.shape[0]//self.batch_size}  bytes/batch={nbytes_upload}")
            print(f"[Calib] dataset dtype={self.data.dtype} shape={self.data.shape} contiguous={self.data.flags['C_CONTIGUOUS']}")
            print(f"[Calib] upload dtype={self.in_dtype_np} itemsize={upload_itemsize}")

    def get_batch_size(self): 
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > len(self.data):
            if self.verbose:
                print("[Calib] get_batch: None (done)")
            return None

        batch_fp32 = np.ascontiguousarray(self.data[self.idx:self.idx + self.batch_size], dtype=np.float32)
        self.idx += self.batch_size

        # 打印观测（用 fp32，直观）：全局 amax & 每通道 amax（C 维）
        if self.verbose:
            amax_global = float(np.max(np.abs(batch_fp32)))
            # 假设 NCHW；对 C 维聚合
            amax_per_ch = np.max(np.abs(batch_fp32), axis=(0, 2, 3))
            print(f"[Calib] get_batch called for inputs={list(names)}")
            print(f"[Calib] get_batch idx=[{self.idx - self.batch_size}:{self.idx}) shape={batch_fp32.shape}")
            print(f"[Calib] batch amax_global={amax_global:.5g}")
            if amax_per_ch.ndim == 1:
                preview = np.array2string(amax_per_ch[:min(10, amax_per_ch.size)], precision=4, separator=' ')
                print(f"[Calib] batch amax_per_channel(first 10)={preview}")

        # —— 上传用期望 dtype —— #
        batch_upload = np.ascontiguousarray(batch_fp32.astype(self.in_dtype_np, copy=False))
        src = batch_upload.view(np.uint8)
        self.h_page[:src.nbytes] = src.ravel()
        cuda.memcpy_htod_async(self.d_in, self.h_page, self.stream)
        self.stream.synchronize()  # 避免异步竞态
        return [int(self.d_in)]

    # cache I/O
    def read_calibration_cache(self):
        if (not self.force) and os.path.exists(self.cache_file):
            print(f"[Calib] load cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[Calib] wrote cache: {self.cache_file}")


class NPZMinMaxCalibrator(_BaseNPZCalibrator, trt.IInt8MinMaxCalibrator):
    def __init__(self, *args, **kwargs):
        trt.IInt8MinMaxCalibrator.__init__(self)
        _BaseNPZCalibrator.__init__(self, *args, **kwargs)

class NPZEntropyCalibrator(_BaseNPZCalibrator, trt.IInt8EntropyCalibrator2):
    def __init__(self, *args, **kwargs):
        trt.IInt8EntropyCalibrator2.__init__(self)
        _BaseNPZCalibrator.__init__(self, *args, **kwargs)

# ----------------- Build & Export -----------------
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

def build_int8_engine(onnx, engine_path, npz_path, shape, ws_mb, algo, cache_path,
                      txt_path, json_path, force=False, max_calib_batches=None, sanity=0, verbose=False):  # >>> CHANGED

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser  = trt.OnnxParser(network, logger)
    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: parse onnx failed")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(1)

    # >>> NEW: Verbose ONNX / network info
    vprint(verbose, f"[Build] Parsed ONNX: {onnx}")
    vprint(verbose, f"[Build] num_inputs={network.num_inputs}, num_outputs={network.num_outputs}, num_layers={network.num_layers}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        vprint(verbose, f"[Build] Input[{i}]: name={inp.name} shape={tuple(inp.shape)} dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        vprint(verbose, f"[Build] Output[{i}]: name={out.name} shape={tuple(out.shape)} dtype={out.dtype}")

    inp = network.get_input(0)
    print(f"[Build] input: {inp.name}  onnx-shape: {tuple(inp.shape)}")

    # 配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws_mb) << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    vprint(verbose, f"[Build] workspace={ws_mb} MB  algo={algo}  INT8=on")

    # profile（固定 NCHW）
    B,C,H,W = shape
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (B,C,H,W), (B,C,H,W), (B,C,H,W))
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)
    vprint(verbose, f"[Build] opt profile fixed to {(B,C,H,W)}")
    
    # 显式打印 input 绑定期望的 dtype（通常是 fp32），与 npz 的 dtype 对齐
    for i in range(network.num_inputs):
        ii = network.get_input(i)
        vprint(verbose, f"[Build] Input[{i}] name={ii.name} expect_shape={tuple(ii.shape)} expect_dtype={ii.dtype}")

    # 选择校准器
    Calib = NPZMinMaxCalibrator if algo == "minmax" else NPZEntropyCalibrator
    in_dtype_trt = inp.dtype  # e.g., trt.DataType.HALF
    in_dtype_np  = trt_dtype_to_np(in_dtype_trt)
    if verbose:
        print(f"[Build] Input[0] name={inp.name} expect_shape={tuple(inp.shape)} expect_dtype={in_dtype_trt}")
        print(f"[Build] Upload will use numpy dtype: {in_dtype_np}")

    calib = Calib(npz_path, batch_size=B, cache_file=cache_path,
              max_calib_batches=max_calib_batches, force=force,
              in_dtype_np=in_dtype_np, verbose=verbose)
    config.int8_calibrator = calib
    print(f"[Build] Using calibrator: {Calib.__name__}")

    # 构建（触发校准）
    print("[Build] Building INT8 engine ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[OK] engine: {engine_path}")
    print(f"[OK] cache : {cache_path}")

    # >>> NEW: parse cache -> true scales -> export with ranges_dict
    try:
        scales = parse_trt_calib_cache_to_scales(cache_path, verbose=verbose)
    except Exception as e:
        print(f"[Warn] parse calib cache failed, fallback to scale=1.0: {e}")
        scales = None

    header = f"TRT-INT8-{('MinMax' if algo=='minmax' else 'EntropyCalibration2')}"
    export_txt_json_from_profile(network, header, txt_path, json_path, ranges_dict=scales)  # >>> CHANGED: pass scales

    # 可选 sanity：用少量样本对比 INT8 与 FP32 logits 余弦相似度
    if sanity and sanity > 0:
        try:
            sanity_compare_fp32_vs_int8(onnx, engine_path, npz_path, B, C, H, W, sanity, verbose=verbose)  # >>> CHANGED
        except Exception as e:
            print(f"[Warn] sanity compare failed: {e}")

# ----------------- Sanity compare (optional) -----------------
def trt_run(engine_path, X):
    """
    执行给定 TensorRT engine，严格按 binding 的真实 dtype/shape 分配显存和 host 内存，
    然后把输出安全地转成 float32 返回。
    """
    import pycuda.driver as cuda
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)

    # 反序列化
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine failed")

    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError("create_execution_context failed")

    # 绑定索引
    in_idx  = next(i for i in range(engine.num_bindings) if engine.binding_is_input(i))
    out_idx = next(i for i in range(engine.num_bindings) if not engine.binding_is_input(i))

    # 设置形状（动态时）
    in_shape_engine = tuple(engine.get_binding_shape(in_idx))
    if in_shape_engine[0] == -1:
        ctx.set_binding_shape(in_idx, X.shape)

    # 真实 dtype
    in_dtype_trt  = engine.get_binding_dtype(in_idx)
    out_dtype_trt = engine.get_binding_dtype(out_idx)
    in_dtype_np   = trt.nptype(in_dtype_trt)   # e.g. np.float16
    out_dtype_np  = trt.nptype(out_dtype_trt)  # e.g. np.float16

    # 类型对齐（必要时转换到 engine 期望的输入 dtype）
    X_in = np.ascontiguousarray(X.astype(in_dtype_np, copy=False))

    # 输出形状（来自 context）
    out_shape = tuple(ctx.get_binding_shape(out_idx))
    if any(d <= 0 for d in out_shape):
        raise RuntimeError(f"invalid output shape from context: {out_shape}")

    # 分配显存
    d_in  = cuda.mem_alloc(X_in.nbytes)
    d_out = cuda.mem_alloc(int(np.prod(out_shape)) * np.dtype(out_dtype_np).itemsize)

    # 绑定数组
    bindings = [0] * engine.num_bindings
    bindings[in_idx]  = int(d_in)
    bindings[out_idx] = int(d_out)

    # 拷贝 + 执行 + 拷回（同步版，确保稳定）
    cuda.memcpy_htod(d_in, X_in)
    ok = ctx.execute_v2(bindings)
    if not ok:
        d_in.free(); d_out.free()
        raise RuntimeError("execute_v2 failed")

    h_out = np.empty(out_shape, dtype=out_dtype_np)
    cuda.memcpy_dtoh(h_out, d_out)

    d_in.free(); d_out.free()

    # 统一转成 float32，便于后续数值比较
    return h_out.astype(np.float32, copy=False)

def cosine_sim(a, b, eps=1e-9):
    """
    稳健版余弦：先把 NaN/Inf 压干净，再算平均余弦。
    """
    a = a.reshape(a.shape[0], -1).astype(np.float32, copy=False)
    b = b.reshape(b.shape[0], -1).astype(np.float32, copy=False)

    # 把 inf/NaN 变成有限值，避免溢出污染
    a = np.nan_to_num(a, posinf=1e30, neginf=-1e30)
    b = np.nan_to_num(b, posinf=1e30, neginf=-1e30)

    na = np.linalg.norm(a, axis=1) + eps
    nb = np.linalg.norm(b, axis=1) + eps
    cs = np.sum(a * b, axis=1) / (na * nb)

    # 筛掉异常的 NaN（理论上不会再有）
    cs = cs[np.isfinite(cs)]
    if cs.size == 0:
        return float("nan")
    return float(np.mean(cs))


def sanity_compare_fp32_vs_int8(onnx, int8_engine, npz, B, C, H, W, N=32, verbose=False):  # >>> CHANGED
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser  = trt.OnnxParser(network, logger)
    with open(onnx, "rb") as f:
        assert parser.parse(f.read())
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, (B,C,H,W),(B,C,H,W),(B,C,H,W))
    config.add_optimization_profile(profile)
    fp32_ser = builder.build_serialized_network(network, config)
    fp32_path = os.path.splitext(int8_engine)[0] + ".fp32.tmp.engine"
    with open(fp32_path, "wb") as f:
        f.write(fp32_ser)

    d = np.load(npz)["imgs"].astype(np.float32, copy=False)
    d = d[:N]
    M = (N//B)*B
    if M == 0:
        raise ValueError("sanity N too small for batch")
    x = d[:M]
    y_int8 = trt_run(int8_engine, x)
    y_fp32 = trt_run(fp32_path, x)
    if y_int8.ndim > 2: y_int8 = y_int8.reshape(y_int8.shape[0], -1)
    if y_fp32.ndim > 2: y_fp32 = y_fp32.reshape(y_fp32.shape[0], -1)
    cs = cosine_sim(y_int8, y_fp32)
    print(f"[Sanity] CosineSim(INT8 vs FP32) on {M} samples: {cs:.4f} (>=0.95 正常；<0.7 多半是早期崩)")
    os.remove(fp32_path)
    vprint(verbose, f"[Sanity] done. tmp fp32 engine removed: {fp32_path}")

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser("Robust INT8 calibrate & export")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--npz",  required=True, help="calibration npz with imgs (already normalized)")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--cache",  required=True)
    ap.add_argument("--calib-txt", required=True)
    ap.add_argument("--calib-json", required=True)
    ap.add_argument("--shape", default="1x3x32x32")
    ap.add_argument("--ws", type=int, default=1024)
    ap.add_argument("--algo", choices=["minmax","entropy"], default="minmax")
    ap.add_argument("--force", action="store_true", help="ignore existing cache; force fresh calibration")
    ap.add_argument("--max-calib-batches", type=int, default=None,
                    help="full batches used for calibration; None or <=0 means no limit")
    ap.add_argument("--sanity", type=int, default=0, help="compare INT8 vs FP32 logits on N samples")
    ap.add_argument("--verbose", action="store_true", help="print detailed build/calib logs")  # >>> NEW
    return ap.parse_args()

def main():
    args = parse_args()
    B,C,H,W = tuple(map(int, args.shape.lower().split("x")))
    vprint(args.verbose, f"[Args] onnx={args.onnx}")
    vprint(args.verbose, f"[Args] npz={args.npz}")
    vprint(args.verbose, f"[Args] engine={args.engine} cache={args.cache}")
    vprint(args.verbose, f"[Args] calib-txt={args.calib_txt} calib-json={args.calib_json}")
    vprint(args.verbose, f"[Args] shape={(B,C,H,W)} ws={args.ws} algo={args.algo} force={args.force} sanity={args.sanity}")

    build_int8_engine(onnx=args.onnx,
                      engine_path=args.engine,
                      npz_path=args.npz,
                      shape=(B,C,H,W),
                      ws_mb=args.ws,
                      algo=args.algo,
                      cache_path=args.cache,
                      txt_path=args.calib_txt,
                      json_path=args.calib_json,
                      force=args.force,
                      max_calib_batches=args.max_calib_batches,
                      sanity=args.sanity,
                      verbose=args.verbose)  # >>> CHANGED

if __name__ == "__main__":
    main()

