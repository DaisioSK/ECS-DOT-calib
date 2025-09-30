import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  # 初始化 CUDA 上下文，便于快速跑 INT8/FP32 对比
from src.utils.common import vprint, cosine_sim


def trt_run(engine_path, X):
    """
    执行给定 TensorRT engine，严格按 binding 的真实 dtype/shape 分配显存和 host 内存，
    然后把输出安全地转成 float32 返回。
    """
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


def sanity_compare_fp32_vs_int8(onnx, int8_engine, npz, B, C, H, W, N=32, verbose=False):  # >>> CHANGED

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
