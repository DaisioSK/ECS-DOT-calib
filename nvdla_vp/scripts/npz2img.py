#!/usr/bin/env python3
# scripts/npz2img.py
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def pick_images_array(npz):
    """
    从 npz 中挑出最像图像数组的那个：
    - 4 维
    - 最后一维为 3（RGB）或 1（灰度）
    - 或者 3 维 (H,W,C) / (C,H,W)
    优先常见键名：images, x_test, X_test, data, x
    """
    prefer = ["images", "x_test", "X_test", "data", "x"]
    keys = list(npz.keys())
    cand_keys = [k for k in prefer if k in keys] + [k for k in keys if k not in prefer]
    for k in cand_keys:
        arr = npz[k]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim == 4:  # (N,H,W,C) or (N,C,H,W)
            return arr
        if arr.ndim == 3:  # 单张 (H,W,C)/(C,H,W)，包一层 N
            return arr[None, ...]
    raise ValueError(f"找不到图像数组，keys={keys}")

def to_hwc_uint8(img):
    """
    接受形状 (H,W,C) 或 (C,H,W) 或 (H,W)；转为 HWC 的 uint8。
    对 float 输入：若最大值<=1.0，按 0-1 缩放到 0-255；否则按 0-255 裁剪。
    """
    arr = img
    # 调整维度到 HWC
    if arr.ndim == 3:
        if arr.shape[0] in (1,3) and arr.shape[-1] not in (1,3):
            # CHW -> HWC
            arr = np.transpose(arr, (1,2,0))
    elif arr.ndim == 2:
        # 灰度 -> HWC
        arr = arr[:, :, None]
    else:
        raise ValueError(f"不支持的图像维度: {arr.shape}")

    # dtype 归一化到 uint8
    if np.issubdtype(arr.dtype, np.floating):
        maxv = float(np.nanmax(arr))
        scale = 255.0 if maxv <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.integer):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    # 如果是单通道，扩展成 3 通道（有些 runtime 期望 RGB）
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr

def main():
    ap = argparse.ArgumentParser(description="Extract N images from NPZ and save as JPGs.")
    ap.add_argument("--npz", required=True, help="输入的 npz 路径（与模型同名）")
    ap.add_argument("--count", type=int, required=True, help="导出图片数量")
    ap.add_argument("--out-root", default="data/img", help="输出根目录（默认 data/img）")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ 不存在: {npz_path}")

    base = npz_path.stem  # 用于目录名/文件名前缀
    out_dir = Path(args.out_root) / base
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(npz_path, allow_pickle=True)
    imgs = pick_images_array(z)

    # 期望 N,H,W,C（RGB=3）
    N = imgs.shape[0] if imgs.ndim >= 3 else 1
    take = max(0, min(args.count, N))
    if take == 0:
        raise ValueError(f"可用图片数量为 0（N={N}）")

    saved = 0
    for i in range(take):
        hwc = to_hwc_uint8(imgs[i])
        im = Image.fromarray(hwc)
        fn = f"{base}_{i+1:04d}.jpg"
        # im.save((out_dir / fn).with_suffix(".ppm"))
        im.save(out_dir / fn, quality=95)
        saved += 1

    print(f"✅ 生成 {saved} 张图片：{out_dir}")

if __name__ == "__main__":
    main()
