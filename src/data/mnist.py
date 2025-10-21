# src/data/providers/mnist.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np, random
from pathlib import Path

DEFAULTS = {
    "calib_count": 1024,
    "eval_count": 10000,   # MNIST test 全量
    "seed": 42,
    "scale": 1.0/255.0,
    "mean": 0.1307,
    "std":  0.3081,
}

@dataclass
class MnistProvider:
    root: str = "data/_cache"  # torchvision 下载缓存目录

    def import_dependencies(self):
        try:
            import torchvision.datasets as tv_datasets
        except ImportError as e:
            raise RuntimeError(
                "MNIST data provider needs torchvision (and torch). "
                "Run: pip install --upgrade torch torchvision"
            ) from e

        try:
            from PIL import Image as PIL_Image
        except ImportError as e:
            raise RuntimeError(
                "MNIST data provider needs Pillow. "
                "Run: pip install --upgrade pillow"
            ) from e

        # 可选：OpenCV；若不用可移除
        try:
            import cv2
        except ImportError:
            cv2 = None  # 不强制

        # 关键：挂到实例上，prepare() 里用 self.*** 访问
        self._datasets = tv_datasets
        self._PIL_Image = PIL_Image
        self._cv2 = cv2

    def prepare(self, *, path: str, role: str, input_shape: Tuple[int,int,int,int], format: str, **kwargs) -> str:
        
        self.import_dependencies()
        assert format.lower() == "npz", "MNIST provider currently supports only NPZ output."
        B, C_t, H_t, W_t = input_shape  # 目标输入 (batch, channels, height, width)

        # 1) 读取 MNIST（灰度 28x28）
        split = "train" if role == "calib" else "test"
        ds = self._datasets.MNIST(root=self.root, train=(split=="train"), download=True)
        images = ds.data.numpy()       # (N, 28, 28) uint8
        labels = ds.targets.numpy()    # (N,)

        # 2) 采样条数
        count = int(kwargs.get("count", DEFAULTS["calib_count" if role=="calib" else "eval_count"]))
        seed  = int(kwargs.get("seed", DEFAULTS["seed"]))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(images), size=min(count, len(images)), replace=False)
        images = images[idx]
        labels = labels[idx]

        # 3) 预处理：resize → to_channels → scale → normalize → NCHW float32
        scale = float(kwargs.get("scale", DEFAULTS["scale"]))
        mean  = kwargs.get("mean", DEFAULTS["mean"])
        std   = kwargs.get("std",  DEFAULTS["std"])
        # 对于 3 通道情况，复制灰度到 3 个通道，mean/std 同一标量重复三次
        if isinstance(mean, (int, float)): mean = [float(mean)] * int(C_t)
        if isinstance(std,  (int, float)): std  = [float(std)]  * int(C_t)

        out = np.zeros((len(images), C_t, H_t, W_t), dtype=np.float32)

        # 预先构造 mean/std 的广播形状，避免内层 for 循环
        mean_arr = np.array(mean, dtype=np.float32).reshape(C_t, 1, 1)
        std_arr  = np.array(std,  dtype=np.float32).reshape(C_t, 1, 1)
        eps = 1e-12  # 防 0

        for i, img in enumerate(images):
            img_pil = self._PIL_Image.fromarray(img, mode="L")
            img_resized = img_pil.resize((W_t, H_t), self._PIL_Image.BILINEAR)
            f = np.asarray(img_resized, dtype=np.float32) * scale  # (H_t, W_t)
            if C_t == 1:
                chw = f[None, :, :]                              # (1, H_t, W_t)
            else:
                chw = np.repeat(f[None, :, :], C_t, axis=0)      # (C_t, H_t, W_t)

            chw = (chw - mean_arr) / (std_arr + eps)
            out[i] = chw

        # 4) 落盘（calib 可不强制 labels，但保留有益）
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, imgs=out, labels=labels.astype(np.int64))

        # === 新增：导出一张可视化 JPG（首张样本）===
        jpg_path = str(Path(path).with_suffix(".jpg"))
        vis = out[0]  # (C_t, H_t, W_t)
        # 反变换为 0~255 灰度/彩色（你当前预处理：scale=1/256, mean=0, std=1）
        if vis.shape[0] == 1:
            img255 = np.clip(vis[0] * 256.0, 0, 255).astype(np.uint8)  # (H,W)
            self._PIL_Image.fromarray(img255, mode="L").save(jpg_path)
        else:
            # 若 C_t==3：把 (C,H,W) → (H,W,C)
            img255 = np.clip((np.transpose(vis, (1,2,0)) * 256.0), 0, 255).astype(np.uint8)
            self._PIL_Image.fromarray(img255, mode="RGB").save(jpg_path)

        print(f"[Data] preview jpg saved: {jpg_path}")

        return path

# 注册到全局注册表
from src.core.dataset_registry import register
register("mnist", MnistProvider())
