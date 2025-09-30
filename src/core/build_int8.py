import os, json, struct, argparse, re
import numpy as np
import tensorrt as trt
from src.core.calibrators import NPZMinMaxCalibrator, NPZEntropyCalibrator
from src.io.calib_artifacts import parse_trt_calib_cache_to_scales, export_txt_json_from_profile
from src.utils.common import vprint, trt_dtype_to_np
from src.core.sanity_check import sanity_compare_fp32_vs_int8


def build_int8_engine(onnx, engine_path, npz_path, shape, ws_mb, algo, cache_path,
                      txt_path, json_path, force=False, max_calib_batches=None, sanity=0, verbose=False):  # >>> CHANGED

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser  = trt.OnnxParser(network, logger)
    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: parse onnx failed")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(1)

    # >>> NEW: Verbose ONNX / network info
    vprint(verbose, f"[Build] Parsed ONNX: {onnx}")
    vprint(verbose, f"[Build] num_inputs={network.num_inputs}, num_outputs={network.num_outputs}, num_layers={network.num_layers}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        vprint(verbose, f"[Build] Input[{i}]: name={inp.name} shape={tuple(inp.shape)} dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        vprint(verbose, f"[Build] Output[{i}]: name={out.name} shape={tuple(out.shape)} dtype={out.dtype}")

    inp = network.get_input(0)
    print(f"[Build] input: {inp.name}  onnx-shape: {tuple(inp.shape)}")

    # 配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws_mb) << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    vprint(verbose, f"[Build] workspace={ws_mb} MB  algo={algo}  INT8=on")

    # profile（固定 NCHW）
    B,C,H,W = shape
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (B,C,H,W), (B,C,H,W), (B,C,H,W))
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)
    vprint(verbose, f"[Build] opt profile fixed to {(B,C,H,W)}")
    
    # 显式打印 input 绑定期望的 dtype（通常是 fp32），与 npz 的 dtype 对齐
    for i in range(network.num_inputs):
        ii = network.get_input(i)
        vprint(verbose, f"[Build] Input[{i}] name={ii.name} expect_shape={tuple(ii.shape)} expect_dtype={ii.dtype}")

    # 选择校准器
    Calib = NPZMinMaxCalibrator if algo == "minmax" else NPZEntropyCalibrator
    in_dtype_trt = inp.dtype  # e.g., trt.DataType.HALF
    in_dtype_np  = trt_dtype_to_np(in_dtype_trt)
    if verbose:
        print(f"[Build] Input[0] name={inp.name} expect_shape={tuple(inp.shape)} expect_dtype={in_dtype_trt}")
        print(f"[Build] Upload will use numpy dtype: {in_dtype_np}")

    calib = Calib(npz_path, batch_size=B, cache_file=cache_path,
              max_calib_batches=max_calib_batches, force=force,
              in_dtype_np=in_dtype_np, verbose=verbose)
    config.int8_calibrator = calib
    print(f"[Build] Using calibrator: {Calib.__name__}")

    # 构建（触发校准）
    print("[Build] Building INT8 engine ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[OK] engine: {engine_path}")
    print(f"[OK] cache : {cache_path}")

    # >>> NEW: parse cache -> true scales -> export with ranges_dict
    try:
        scales = parse_trt_calib_cache_to_scales(cache_path, verbose=verbose)
    except Exception as e:
        print(f"[Warn] parse calib cache failed, fallback to scale=1.0: {e}")
        scales = None

    header = f"TRT-INT8-{('MinMax' if algo=='minmax' else 'EntropyCalibration2')}"
    export_txt_json_from_profile(network, header, txt_path, json_path, ranges_dict=scales)  # >>> CHANGED: pass scales

    # 可选 sanity：用少量样本对比 INT8 与 FP32 logits 余弦相似度
    if sanity and sanity > 0:
        try:
            sanity_compare_fp32_vs_int8(onnx, engine_path, npz_path, B, C, H, W, sanity, verbose=verbose)  # >>> CHANGED
        except Exception as e:
            print(f"[Warn] sanity compare failed: {e}")


