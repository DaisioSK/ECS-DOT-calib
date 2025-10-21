# src/data/registry.py
from __future__ import annotations
import importlib
from typing import Dict, Protocol

class Provider(Protocol):
    def prepare(self, *, path: str, role: str, input_shape, format: str, **kwargs) -> str: ...

_REGISTRY: Dict[str, Provider] = {}

def register(name: str, provider: Provider) -> None:
    key = name.strip().lower()
    _REGISTRY[key] = provider

def get_provider(name: str) -> Provider:
    key = name.strip().lower()

    # 直接命中已注册
    if key in _REGISTRY:
        return _REGISTRY[key]
    
    # 动态导入：形如 "pkg.module:factory"
    if ":" in name:
        mod, factory = name.split(":", 1)
        m = importlib.import_module(mod)
        obj = getattr(m, factory)
        prov = obj() if callable(obj) else obj
        return prov
    
    # 根据 source 名按约定尝试导入常见 provider 位置
    for modname in (
        f"src.data.{key}",
    ):
        try:
            importlib.import_module(modname)
            if key in _REGISTRY:
                return _REGISTRY[key]
        except ModuleNotFoundError:
            continue

    raise KeyError(f"Unknown data source '{name}'. Registered: {list(_REGISTRY.keys())}")
