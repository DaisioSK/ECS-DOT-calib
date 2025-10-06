#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single entry pipeline:
1) Calibrate & build INT8 engine from ONNX + calib npz
2) Export calib.{txt,json} (scales)
3) Evaluate engine on eval npz → report.json
Usage:
  python -m src.app.pipeline --cfg configs/cifar10_int8.yaml
"""

import argparse, json
from pathlib import Path
from src.core.build_int8 import build_int8_engine
from src.app.eval import run_eval_on_npz
from src.utils.route import resolve, ensure_dir
from src.utils.common import nowstamp

def load_cfg(p: str) -> dict:
    pth = Path(p)
    if not pth.exists():
        raise FileNotFoundError(p)
    txt = pth.read_text(encoding="utf-8")
    try:
        import yaml  # pip install pyyaml
        return yaml.safe_load(txt) or {}
    except Exception:
        return json.loads(txt)
    
def main():
    ap = argparse.ArgumentParser("INT8 pipeline (single entry)")
    ap.add_argument("--cfg", required=True, help="YAML/JSON config path")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)

    # -------- read config fields --------
    proj = (cfg.get("project") or {})
    model = (cfg.get("model") or {})
    calib = (cfg.get("calibration") or {})
    build = (cfg.get("build") or {})
    evalc = (cfg.get("eval") or {})

    name  = proj.get("name", "model")
    onnx  = resolve(model["onnx"])
    shape = str(model.get("input_shape", "1x3x32x32")).lower()
    ws_mb = int(model.get("workspace_mb", 1024))

    algo   = calib.get("algo", "minmax")
    calib_npz = resolve(calib["npz"])
    max_batches = calib.get("max_batches", None)
    force  = bool(calib.get("force", False))
    sanity = int(calib.get("sanity_n", 0))
    verbose= bool(calib.get("verbose", False))

    engines_out = resolve(build.get("engine_out", "models/engines"))
    calib_out   = resolve(build.get("calib_out",  "models/calib"))
    eval_npz    = resolve(evalc["npz"])
    eval_bs     = int(evalc.get("batch_size", 32))
    reports_out = resolve(evalc.get("outdir", "reports"))

    ensure_dir(engines_out)
    ensure_dir(calib_out)
    ensure_dir(reports_out)

    ts = nowstamp()
    engine_path = engines_out / f"{name}@{shape}@{algo}@{ts}.engine"
    cache_path  = calib_out   / f"{name}.calib.cache"
    txt_path    = calib_out   / f"{name}_calib.txt"
    json_path   = calib_out   / f"{name}_calib.json"

    run_id = f"{name}@int8@{algo}@{shape}@{ts}"
    report_dir = reports_out / run_id
    ensure_dir(report_dir)

    # -------- Stage 1: Calibrate & Build (also exports calib txt/json) --------
    print(f"[Build] onnx={onnx} calib_npz={calib_npz} -> engine={engine_path}")
    B,C,H,W = map(int, shape.split("x"))
    build_int8_engine(
        onnx=str(onnx),
        engine_path=str(engine_path),
        npz_path=str(calib_npz),
        shape=(B,C,H,W),
        ws_mb=ws_mb,
        algo=algo,
        cache_path=str(cache_path),
        txt_path=str(txt_path),
        json_path=str(json_path),
        force=force,
        max_calib_batches=max_batches,
        sanity=sanity,
        verbose=verbose,
    )

    # -------- Stage 2: Evaluate --------
    print(f"[Eval] engine={engine_path} eval_npz={eval_npz} outdir={report_dir}")
    report_paths = run_eval_on_npz(
        engine=str(engine_path),
        npz=str(eval_npz),
        batch_size=eval_bs,
        outdir=str(report_dir),
    )

    # -------- Final: write a tiny meta --------
    meta = {
        "run_id": run_id,
        "project": name,
        "algo": algo,
        "shape": shape,
        "engine": str(engine_path),
        "calib_cache": str(cache_path),
        "calib_txt": str(txt_path),
        "calib_json": str(json_path),
        "eval_report": report_paths.get("report_json"),
        "confusion_matrix": report_paths.get("cm_csv"),
        "config": str(resolve(args.cfg)),
        "created_at": nowstamp(),
    }
    with open(report_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Done] run_id={run_id}")
    print(f"       scales json: {json_path}")
    print(f"       report json: {meta['eval_report']}")

if __name__ == "__main__":
    main()
