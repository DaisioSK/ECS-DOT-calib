import argparse, time, os, json, warnings
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  # 初始化 CUDA 上下文，确保推理前已创建 context

# ---------- TensorRT runner (8.x/9.x 兼容；优先新接口，无警告) ----------
class TrtRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None: raise RuntimeError("Failed to deserialize engine")
        self.context = self.engine.create_execution_context()
        if self.context is None: raise RuntimeError("Failed to create execution context")

        self.has_new_tensor_api = all(
            hasattr(self.engine, name) for name in
            ("get_tensor_name","get_tensor_mode","get_tensor_shape","get_tensor_dtype")
        )
        self.has_enqueue_v3 = hasattr(self.context, "enqueue_v3")

        if self.has_new_tensor_api:
            # 新接口：不会触发 DeprecationWarning
            self.all_tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_names  = [n for n in self.all_tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in self.all_tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            assert len(self.input_names)==1 and len(self.output_names)==1
            self.in_name  = self.input_names[0]
            self.out_name = self.output_names[0]
            self.in_shape_engine = list(self.engine.get_tensor_shape(self.in_name))
            self.fixed_batch = self.in_shape_engine[0] if self.in_shape_engine and self.in_shape_engine[0] != -1 else None
            self.in_dtype  = safe_nptype(self.engine.get_tensor_dtype(self.in_name))
            self.out_dtype = safe_nptype(self.engine.get_tensor_dtype(self.out_name))
            print(f"[TRT] input={self.in_name} shape@engine={self.in_shape_engine} dtype={self.in_dtype}  "
                  f"output={self.out_name} dtype={self.out_dtype}")
        else:
            # 老接口：只在新接口不可用时兜底（避免 warnings）
            self.input_indices  = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
            self.output_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
            assert len(self.input_indices)==1 and len(self.output_indices)==1
            self.in_idx = self.input_indices[0]
            self.out_idx = self.output_indices[0]
            self.in_name = self.engine.get_binding_name(self.in_idx)
            self.out_name= self.engine.get_binding_name(self.out_idx)
            self.in_shape_engine = list(self.engine.get_binding_shape(self.in_idx))
            self.fixed_batch = self.in_shape_engine[0] if self.in_shape_engine and self.in_shape_engine[0] != -1 else None
            self.in_dtype  = safe_nptype(self.engine.get_binding_dtype(self.in_idx))
            self.out_dtype = safe_nptype(self.engine.get_binding_dtype(self.out_idx))
            print(f"[TRT] input={self.in_name} shape@engine={self.in_shape_engine} dtype={self.in_dtype}  "
                  f"output={self.out_name} dtype={self.out_dtype}")

    def _set_shape(self, B, C, H, W):
        """
        动态引擎：设置 (B,C,H,W)
        固定引擎：允许 B <= fixed_batch（B < fixed 时由 infer 内部做 padding），B > fixed 仍然报错
        """
        if self.has_new_tensor_api:
            if self.in_shape_engine[0] == -1:
                # 动态 batch
                self.context.set_input_shape(self.in_name, (B, C, H, W))
            else:
                fixed = self.in_shape_engine[0]
                if B > fixed:
                    raise ValueError(f"Engine fixed batch={fixed}, got larger B={B}")
                # B <= fixed：不改 context 形状，保持 fixed，后续在 infer 里做 pad
        else:
            if self.in_shape_engine[0] == -1:
                # 动态 batch（老 API）
                self.context.set_binding_shape(self.in_idx, (B, C, H, W))
            else:
                fixed = self.in_shape_engine[0]
                if B > fixed:
                    raise ValueError(f"Engine fixed batch={fixed}, got larger B={B}")
                # B <= fixed：不设置 binding 形状，保持 fixed，infer 里做 pad

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        支持两种模式：
          - 动态引擎：按 B 设置形状直接跑
          - 固定引擎：若 B < fixed_batch，自动把输入 pad 到 fixed_batch，再在输出切回前 B 条
        Padding 策略：重复该 batch 的最后一张（更稳妥，避免 out-of-distribution）
        """
        assert x.ndim == 4, f"expect NCHW, got {x.shape}"
        B, C, H, W = x.shape

        # 形状检查/设置（允许 B<=fixed；B>fixed 直接报错）
        self._set_shape(B, C, H, W)

        # ---- 准备输入（必要时 padding）----
        pad_used = False
        if self.fixed_batch is not None and B < self.fixed_batch:
            # 固定 batch，引擎预期 self.fixed_batch；我们补齐
            fixed = self.fixed_batch
            xb = x.astype(self.in_dtype, copy=False)
            x_pad = np.empty((fixed, C, H, W), dtype=self.in_dtype)
            x_pad[:B] = xb
            x_pad[B:] = xb[-1:]     # 重复最后一张
            x_contig = np.ascontiguousarray(x_pad)
            pad_used = True
            B_effective = fixed
        else:
            # 动态引擎，或固定引擎但 B==fixed
            x_contig = np.ascontiguousarray(x.astype(self.in_dtype, copy=False))
            B_effective = B

        # ---- 计算输出 shape 并分配显存 ----
        if self.has_new_tensor_api:
            out_shape_engine = tuple(self.context.get_tensor_shape(self.out_name))
        else:
            out_shape_engine = tuple(self.context.get_binding_shape(self.out_idx))
        # out_shape_engine[0] 可能是 fixed 或 B（动态）；按引擎回报分配
        out_bytes = int(np.prod(out_shape_engine)) * np.dtype(self.out_dtype).itemsize

        d_in = d_out = None
        try:
            d_in  = cuda.mem_alloc(x_contig.nbytes)
            d_out = cuda.mem_alloc(out_bytes)
            cuda.memcpy_htod(d_in, x_contig)

            if self.has_new_tensor_api and self.has_enqueue_v3:
                self.context.set_tensor_address(self.in_name,  int(d_in))
                self.context.set_tensor_address(self.out_name, int(d_out))
                if not self.context.enqueue_v3(stream_handle=0):
                    raise RuntimeError("enqueue_v3 failed")
            else:
                if not self.has_new_tensor_api:
                    bindings = [0]*self.engine.num_bindings
                    bindings[self.in_idx]  = int(d_in)
                    bindings[self.out_idx] = int(d_out)
                else:
                    in_idx  = next(i for i in range(self.engine.num_bindings)
                                   if self.engine.get_binding_name(i)==self.in_name)
                    out_idx = next(i for i in range(self.engine.num_bindings)
                                   if self.engine.get_binding_name(i)==self.out_name)
                    bindings = [0]*self.engine.num_bindings
                    bindings[in_idx]  = int(d_in)
                    bindings[out_idx] = int(d_out)
                if not self.context.execute_v2(bindings):
                    raise RuntimeError("execute_v2 failed")

            h_out = np.empty(out_shape_engine, dtype=self.out_dtype)
            cuda.memcpy_dtoh(h_out, d_out)

        finally:
            if d_in:  d_in.free()
            if d_out: d_out.free()

        # 统一转 float32，必要时切片回真实 B
        y = h_out.astype(np.float32, copy=False)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)

        if pad_used and y.shape[0] >= B:
            y = y[:B]  # 丢弃由 padding 产生的尾部输出
            # 可选提示：只在首次发生时打印
            # print(f"[Eval] Padded batch from {B} to {self.fixed_batch}, outputs sliced back to {B}")

        return y

def safe_nptype(dtype: trt.DataType):
    if dtype == trt.DataType.FLOAT:   return np.float32
    if dtype == trt.DataType.HALF:    return np.float16
    if dtype == trt.DataType.INT8:    return np.int8
    if dtype == trt.DataType.INT32:   return np.int32
    if dtype == trt.DataType.BOOL:    return np.bool_
    if hasattr(trt.DataType, "UINT8") and dtype == trt.DataType.UINT8: return np.uint8
    return np.float32
