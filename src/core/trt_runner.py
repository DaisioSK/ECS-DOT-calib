import numpy as np
from pathlib import Path
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


def safe_nptype(dtype: trt.DataType):
    if dtype == trt.DataType.FLOAT:   return np.float32
    if dtype == trt.DataType.HALF:    return np.float16
    if dtype == trt.DataType.INT8:    return np.int8
    if dtype == trt.DataType.INT32:   return np.int32
    if dtype == trt.DataType.BOOL:    return np.bool_
    if hasattr(trt.DataType, "UINT8") and dtype == trt.DataType.UINT8: return np.uint8
    return np.float32


class TrtRunner:
    """
    同时兼容：
      - 隐式 batch 引擎（Caffe 路径） → context.execute(batch_size)
      - 显式 batch 引擎（ONNX 路径） → context.execute_v2() / enqueue_v3()
    自动选择新/旧 API；对外暴露：
      - is_implicit: 是否隐式 batch
      - fixed_batch: 显式引擎且 batch 固定时的 B；否则 None
      - max_batch_size: 隐式引擎的 builder.max_batch_size；显式下为 None
    """
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # 是否隐式 batch
        self.is_implicit = bool(self.engine.has_implicit_batch_dimension)

        # 能力探测
        self.has_new_tensor_api = (not self.is_implicit) and all(
            hasattr(self.engine, name) for name in
            ("get_tensor_name", "get_tensor_mode", "get_tensor_shape", "get_tensor_dtype")
        )
        self.has_enqueue_v3 = hasattr(self.context, "enqueue_v3")

        if self.is_implicit:
            # 仅使用旧 bindings API
            self.input_indices  = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
            self.output_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
            assert len(self.input_indices) == 1 and len(self.output_indices) == 1
            self.in_idx  = self.input_indices[0]
            self.out_idx = self.output_indices[0]
            self.in_name  = self.engine.get_binding_name(self.in_idx)
            self.out_name = self.engine.get_binding_name(self.out_idx)
            self.in_shape_engine = tuple(self.engine.get_binding_shape(self.in_idx))  # CHW
            self.in_dtype  = safe_nptype(self.engine.get_binding_dtype(self.in_idx))
            self.out_shape_engine = tuple(self.engine.get_binding_shape(self.out_idx))  # 例如 (10,)
            self.out_dtype = safe_nptype(self.engine.get_binding_dtype(self.out_idx))
            self.max_batch_size = getattr(self.engine, "max_batch_size", 1)  # 仅隐式有意义
            self.fixed_batch = None  # 隐式下不暴露 fixed_batch 概念
            print(f"[TRT] (implicit) input={self.in_name} shape@engine={self.in_shape_engine} dtype={self.in_dtype}  "
                  f"output={self.out_name} shape@engine={self.out_shape_engine} dtype={self.out_dtype}  "
                  f"max_batch_size={self.max_batch_size}")

        else:
            # 显式 batch：优先新 Tensor API，兜底旧 bindings API
            if self.has_new_tensor_api:
                self.all_tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
                self.input_names  = [n for n in self.all_tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
                self.output_names = [n for n in self.all_tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
                assert len(self.input_names) == 1 and len(self.output_names) == 1
                self.in_name  = self.input_names[0]
                self.out_name = self.output_names[0]
                self.in_shape_engine = tuple(self.engine.get_tensor_shape(self.in_name))  # 可能含 -1
                self.in_dtype  = safe_nptype(self.engine.get_tensor_dtype(self.in_name))
                self.out_dtype = safe_nptype(self.engine.get_tensor_dtype(self.out_name))
                self.fixed_batch = self.in_shape_engine[0] if (len(self.in_shape_engine) > 0 and self.in_shape_engine[0] != -1) else None
                self.max_batch_size = None
                print(f"[TRT] (explicit/new) input={self.in_name} shape@engine={self.in_shape_engine} dtype={self.in_dtype}  "
                      f"output={self.out_name} dtype={self.out_dtype} fixed_batch={self.fixed_batch}")
            else:
                self.input_indices  = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
                self.output_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
                assert len(self.input_indices) == 1 and len(self.output_indices) == 1
                self.in_idx  = self.input_indices[0]
                self.out_idx = self.output_indices[0]
                self.in_name  = self.engine.get_binding_name(self.in_idx)
                self.out_name = self.engine.get_binding_name(self.out_idx)
                self.in_shape_engine = tuple(self.engine.get_binding_shape(self.in_idx))  # 可能含 -1
                self.in_dtype  = safe_nptype(self.engine.get_binding_dtype(self.in_idx))
                self.out_dtype = safe_nptype(self.engine.get_binding_dtype(self.out_idx))
                self.fixed_batch = self.in_shape_engine[0] if (len(self.in_shape_engine) > 0 and self.in_shape_engine[0] != -1) else None
                self.max_batch_size = None
                print(f"[TRT] (explicit/old) input={self.in_name} shape@engine={self.in_shape_engine} dtype={self.in_dtype}  "
                      f"output={self.out_name} dtype={self.out_dtype} fixed_batch={self.fixed_batch}")

        # 简单的设备缓冲（每次 infer 分配/释放；需要更快可做缓存池）
        self._d_in = None
        self._d_out = None

    # ----------------------------
    # 推理
    # ----------------------------
    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        x: NCHW float32/float16 contiguous
        返回：二维 logits（N, K）或原始形状可 reshape 后处理
        """
        assert x.ndim == 4, f"Expect NCHW, got {x.shape}"
        x = np.ascontiguousarray(x, dtype=self.in_dtype)
        B, C, H, W = x.shape

        if self.is_implicit:
            # --- 隐式 batch：使用 execute(batch_size) ---
            if self.max_batch_size and B > self.max_batch_size:
                raise ValueError(f"Batch {B} > engine.max_batch_size {self.max_batch_size}")
            c, h, w = self.in_shape_engine  # CHW
            in_elems  = B * c * h * w
            out_elems = B * int(np.prod(self.out_shape_engine))

            # 分配显存
            d_in  = cuda.mem_alloc(x.nbytes)
            d_out = cuda.mem_alloc(out_elems * np.dtype(self.out_dtype).itemsize)
            try:
                cuda.memcpy_htod(d_in, x)
                bindings = [0] * self.engine.num_bindings
                bindings[self.in_idx]  = int(d_in)
                bindings[self.out_idx] = int(d_out)
                ok = self.context.execute(B, bindings)
                if not ok:
                    raise RuntimeError("context.execute failed (implicit-batch)")
                h_out = np.empty(out_elems, dtype=self.out_dtype)
                cuda.memcpy_dtoh(h_out, d_out)
                y = h_out.reshape(B, -1).astype(np.float32, copy=False)
                return y
            finally:
                d_in.free(); d_out.free()

        # --- 显式 batch：设置形状并执行 ---
        # 计算输出 shape 使用 context 获取（显式时才有意义）
        if self.has_new_tensor_api:
            # 取引擎声明的输入 shape（可能含 -1；固定 batch 时首维是正数）
            shp = tuple(self.engine.get_tensor_shape(self.in_name))
            is_dynamic = (len(shp) > 0 and shp[0] == -1)
            fixed = self.fixed_batch  # None 或 正数

            # ====== 关键：显式固定 batch 的 padding 逻辑 ======
            pad_used = False
            if (not is_dynamic) and (fixed is not None) and (B < fixed):
                xb = x  # 已是 contiguous & 正确 dtype
                x_pad = np.empty((fixed, C, H, W), dtype=self.in_dtype)
                x_pad[:B] = xb
                x_pad[B:] = xb[-1:]         # 重复最后一张
                x_eff = x_pad
                B_eff = fixed
                pad_used = True
            else:
                # 动态 batch 或 B==fixed
                if is_dynamic:
                    # 动态批次：显式设置当次输入形状
                    self.context.set_input_shape(self.in_name, (B, C, H, W))
                x_eff = x
                B_eff = B

            # 推断输出元素个数（按 context 的当前 shape）
            out_shape_ctx = tuple(self.context.get_tensor_shape(self.out_name))
            # 有的版本此时仍含 -1；按执行后拉平再以 B_eff 还原类别数更稳
            in_bytes  = x_eff.nbytes
            # 先按 shape 估计一个上界；执行后我们再 reshape
            out_elems_est = int(np.prod([d if d > 0 else B_eff for d in out_shape_ctx]))
            out_bytes = out_elems_est * np.dtype(self.out_dtype).itemsize

            d_in  = cuda.mem_alloc(in_bytes)
            d_out = cuda.mem_alloc(out_bytes)
            try:
                cuda.memcpy_htod(d_in, x_eff)

                if self.has_enqueue_v3:
                    self.context.set_tensor_address(self.in_name,  int(d_in))
                    self.context.set_tensor_address(self.out_name, int(d_out))
                    if not self.context.enqueue_v3(stream_handle=0):
                        raise RuntimeError("enqueue_v3 failed")
                else:
                    # 兜底 execute_v2：把 tensor name 映射到 binding 索引
                    in_idx = next(i for i in range(self.engine.num_bindings)
                                if self.engine.get_binding_name(i) == self.in_name)
                    out_idx = next(i for i in range(self.engine.num_bindings)
                                if self.engine.get_binding_name(i) == self.out_name)
                    bindings = [0] * self.engine.num_bindings
                    bindings[in_idx]  = int(d_in)
                    bindings[out_idx] = int(d_out)
                    if not self.context.execute_v2(bindings):
                        raise RuntimeError("execute_v2 failed (explicit/new)")

                # 执行完后，再以 B_eff 复原 (B_eff, K)
                # 注意：某些版本的 get_tensor_shape 执行后仍不更新，这里直接用 bytes 拉平
                h_out = np.empty(out_elems_est, dtype=self.out_dtype)
                cuda.memcpy_dtoh(h_out, d_out)
                y = h_out.reshape(B_eff, -1).astype(np.float32, copy=False)
                if pad_used:
                    y = y[:B]  # 切回真实 batch
                return y
            finally:
                d_in.free(); d_out.free()
        else:
            # 显式旧 API：set_binding_shape + execute_v2
            if self.in_shape_engine and self.in_shape_engine[0] == -1:
                self.context.set_binding_shape(self.in_idx, (B, C, H, W))
            in_bytes = x.nbytes
            out_shape_ctx = tuple(self.context.get_binding_shape(self.out_idx))
            out_elems = int(np.prod(out_shape_ctx))
            out_bytes = out_elems * np.dtype(self.out_dtype).itemsize

            d_in  = cuda.mem_alloc(in_bytes)
            d_out = cuda.mem_alloc(out_bytes)
            try:
                cuda.memcpy_htod(d_in, x)
                bindings = [0] * self.engine.num_bindings
                bindings[self.in_idx]  = int(d_in)
                bindings[self.out_idx] = int(d_out)
                if not self.context.execute_v2(bindings):
                    raise RuntimeError("execute_v2 failed (explicit/old)")
                h_out = np.empty(out_elems, dtype=self.out_dtype)
                cuda.memcpy_dtoh(h_out, d_out)
                return h_out.reshape(B, -1).astype(np.float32, copy=False)
            finally:
                d_in.free(); d_out.free()
