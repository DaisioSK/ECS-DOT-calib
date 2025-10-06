from pathlib import Path

def resolve(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (Path(".").resolve() / p).resolve()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

