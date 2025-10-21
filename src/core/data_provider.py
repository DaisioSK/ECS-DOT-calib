# src/data/provider_api.py
from __future__ import annotations
import os, json, hashlib
from pathlib import Path
from typing import Dict, Any, Tuple
from src.core.dataset_registry import get_provider
try:
    # 复用你的 ensure_dir（若不存在则本地兜底）
    from src.utils.route import ensure_dir
except Exception:
    def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def _fingerprint(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]

def _write_meta(npz_path: str, meta: Dict[str, Any]) -> None:
    meta_path = Path(npz_path).with_suffix(Path(npz_path).suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def _load_meta(npz_path: str) -> Dict[str, Any] | None:
    meta_path = Path(npz_path).with_suffix(Path(npz_path).suffix + ".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def prepare_data_if_needed(data_cfg: Dict[str, Any], role: str, input_shape: Tuple[int,int,int,int]) -> str:
    """
    data_cfg: {
      "format": "npz",
      "path": "data/calib/mnist.npz",
      "source": "mnist" | "pkg.module:factory" | None,
      "source_args": {...}  # 可选，覆盖 provider 默认
    }
    role: "calib" | "eval"
    input_shape: (B,C,H,W) —— 用于推断 resize / 通道
    """
    fmt   = (data_cfg.get("format") or "npz").lower()
    path  = data_cfg.get("path")
    source= data_cfg.get("source")
    sargs = data_cfg.get("source_args") or {}

    if not path:
        # 若用户没给 path，按约定自动落地
        default_dir = "data/calib" if role == "calib" else "data/eval"
        default_name = (source or "data") + ".npz"
        path = str(Path(default_dir) / default_name)
        data_cfg["path"] = path

    # 1) 直通：文件已存在 → 直接使用
    if Path(path).exists():
        return path

    # 2) 需要采集：必须有 source
    if not source:
        raise FileNotFoundError(f"Data file not found at '{path}', and no 'source' specified to generate it.")

    # 3) 调用 provider 准备数据
    provider = get_provider(source)
    ensure_dir(Path(path).parent)
    meta_req = {"role": role, "format": fmt, "input_shape": input_shape, "source": source, "source_args": sargs}
    meta_fpr = _fingerprint(meta_req)

    # 命中缓存（同配置下已生成过且 meta 一致）就直接返回
    old = _load_meta(path)
    if old and old.get("_fingerprint") == meta_fpr and Path(path).exists():
        return path

    out_path = provider.prepare(path=path, role=role, input_shape=input_shape, format=fmt, **sargs)
    # 写 meta
    meta_out = {
        **meta_req,
        "_fingerprint": meta_fpr,
        "generated_path": out_path,
    }
    _write_meta(out_path, meta_out)
    return out_path
