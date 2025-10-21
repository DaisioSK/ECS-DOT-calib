import time
import numpy as np

def nowstamp() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S")

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
    
def normalize_shape(s):
    """
    Accepts: "1x3x32x32" / "1,3,32,32" / [1,3,32,32] / (1,3,32,32)
    Returns: (1,3,32,32)
    """
    if isinstance(s, (list, tuple)):
        if len(s) != 4: raise ValueError(f"input_shape length must be 4, got {len(s)}: {s}")
        return tuple(int(x) for x in s)
    if isinstance(s, str):
        t = s.lower().replace(",", "x").strip()
        parts = [p for p in t.split("x") if p]
        if len(parts) != 4:
            raise ValueError(f"input_shape must have 4 dims, got: {s}")
        return tuple(int(p) for p in parts)
    raise TypeError(f"Unsupported input_shape type: {type(s)}")
