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
from src.app.eval import run_eval
from src.utils.route import resolve, ensure_dir
from src.utils.common import nowstamp, normalize_shape
from src.core.data_provider import prepare_data_if_needed

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
    model_format =(model.get("format") or {})
    model_params =(model.get("params") or {})

    input_name = str(model_params.get("input_name", "input"))
    ws_mb = int(model_params.get("workspace_mb", 1024))
    shape = str(model_params.get("input_shape", "1x3x32x32")).lower()
    shape_bchw = normalize_shape(shape)

    if model_format == "onnx":
        onnx  = resolve(model_params["path"])
        
    elif model_format == "caffe":
        prototxt = str(resolve(model_params["prototxt"]))
        caffe = str(resolve(model_params["caffemodel"]))

        bgr = bool(model_params.get("bgr", True))
        pretrn_mean = list(model_params["mean"])
        pretrn_std = list(model_params["std"])


    # Stage 0: 准备数据（若 path 不存在且配置了 source，则自动生成）
    calib_data_path = prepare_data_if_needed((calib.get("data") or {}), role="calib", input_shape=shape_bchw)
    eval_data_path  = prepare_data_if_needed((evalc.get("data") or {}), role="eval", input_shape=shape_bchw)
    

    # calib_data = (calib.get("data") or {})
    # calib_data_format = str(calib_data.get("format") or "npz")
    # if calib_data_format == "npz":
    #     calib_data_path = resolve(calib_data["path"])

    algo   = calib.get("algo", "minmax")
    max_batches = calib.get("max_batches", None)
    force  = bool(calib.get("force", False))
    sanity = int(calib.get("sanity_n", 0))
    verbose= bool(calib.get("verbose", False))

    engines_out = resolve(build.get("engine_out", "models/engines"))
    calib_out   = resolve(build.get("calib_out",  "models/calib"))

    eval_data = (evalc.get("data") or {})
    eval_data_format = str(eval_data.get("format") or "npz")
    # if eval_data_format == "npz":
    #     eval_data_path = resolve(eval_data["path"])
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
    # build_int8_engine(format, model_path, engine_path, calib_data_path, config={}
    if model_format == "onnx":
        print(f"[Build] onnx={onnx} calib_npz={calib_data_path} -> engine={engine_path}")
        calib_args = {
            "onnx":str(onnx),
            "engine_path":str(engine_path),
            "npz_path":str(calib_data_path),
            "shape":shape_bchw,
            "ws_mb":ws_mb,
            "algo":algo,
            "cache_path":str(cache_path),
            "txt_path":str(txt_path),
            "json_path":str(json_path),
            "force":force,
            "max_calib_batches":max_batches,
            "sanity":sanity,
            "verbose":verbose
        }
    elif model_format == "caffe":
        calib_args = dict(
            prototxt=prototxt,
            caffemodel=caffe,
            input_name=input_name,
            shape=shape_bchw,
            ws_mb=ws_mb,
            algo=algo,
            verbose=verbose,
            npz_path=str(calib_data_path),
            engine_path=str(engine_path),
            cache_path=str(cache_path),
            txt_path=str(txt_path),
            json_path=str(json_path),
        )

        # bgr = bool(model_params.get("bgr", True))
        # pretrn_mean = list(model_params["mean"])
        # pretrn_std = list(model_params["std"])


    build_int8_engine(model_format, calib_args)

    # -------- Stage 2: Evaluate --------
    if eval_data_format == "npz":
        print(f"[Eval] engine={engine_path} eval_npz={eval_data_path} outdir={report_dir}")
        eval_args = {
            "npz":str(eval_data_path),
            "batch_size":eval_bs,
            "outdir":str(report_dir),
        }

    report_paths = run_eval(eval_data_format, engine_path, eval_args)

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
