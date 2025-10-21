import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  # 初始化 CUDA 上下文，供校准批量上传使用

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

        z = np.load(npz_path)
        key = "imgs" if "imgs" in z.files else "images"
        arr = z[key]
        if arr.ndim != 4:
            raise ValueError(f"npz imgs must be [N,C,H,W], got {arr.shape}")
        
        # convert dtype to float32 for data distribution stats. will be converted back to in_dtype_np before upload
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        # finalize samples
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

        # calculate bytes per batch upload, and assign pinned + device accordingly
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

        # verbose review: global amax and amax per channel
        if self.verbose:
            amax_global = float(np.max(np.abs(batch_fp32)))
            amax_per_ch = np.max(np.abs(batch_fp32), axis=(0, 2, 3))
            print(f"[Calib] get_batch called for inputs={list(names)}")
            print(f"[Calib] get_batch idx=[{self.idx - self.batch_size}:{self.idx}) shape={batch_fp32.shape}")
            print(f"[Calib] batch amax_global={amax_global:.5g}")
            if amax_per_ch.ndim == 1:
                preview = np.array2string(amax_per_ch[:min(10, amax_per_ch.size)], precision=4, separator=' ')
                print(f"[Calib] batch amax_per_channel(first 10)={preview}")

        # cast FP32 (for stats) -> model input dtype; then take uint8 view for raw byte copy only
        batch_upload = np.ascontiguousarray(batch_fp32.astype(self.in_dtype_np, copy=False))
        src = batch_upload.view(np.uint8)
        self.h_page[:src.nbytes] = src.ravel()

        # async copy data from CPU to VRAM
        cuda.memcpy_htod_async(self.d_in, self.h_page, self.stream)
        self.stream.synchronize()
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
