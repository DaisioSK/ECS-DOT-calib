import os, json, struct, argparse, re
import numpy as np
import tensorrt as trt
from src.core.calibrators import NPZMinMaxCalibrator, NPZEntropyCalibrator
from src.io.calib_artifacts import parse_trt_calib_cache_to_scales, export_txt_json_from_profile
from src.utils.common import vprint, trt_dtype_to_np
from src.core.sanity_check import sanity_compare_fp32_vs_int8

def build_int8_engine(format, args={}):
    if format == "onnx":
        required = ["onnx","engine_path","npz_path","shape","ws_mb","algo",
                    "cache_path","txt_path","json_path"]
        _ensure_args(required, args)
        return build_int8_engine_onnx(
            onnx=args["onnx"],
            engine_path=args["engine_path"],
            npz_path=args["npz_path"],
            shape=args["shape"],                   
            ws_mb=args["ws_mb"],
            algo=args["algo"],
            cache_path=args["cache_path"],
            txt_path=args["txt_path"],
            json_path=args["json_path"],
            force=args.get("force", False),
            max_calib_batches=args.get("max_calib_batches"),
            sanity=args.get("sanity", 0),
            verbose=args.get("verbose", False),
        )

    elif format == "caffe":
        required = ["prototxt","caffemodel","input_name",
                    "engine_path","npz_path","shape","ws_mb","algo",
                    "cache_path","txt_path","json_path"]
        _ensure_args(required, args)
        
        return build_int8_engine_caffe(
            prototxt=args["prototxt"],
            caffemodel=args["caffemodel"],
            input_name=args["input_name"],
            engine_path=args["engine_path"],
            npz_path=args["npz_path"],
            shape=args["shape"],
            ws_mb=args["ws_mb"],
            algo=args["algo"],
            cache_path=args["cache_path"],
            txt_path=args["txt_path"],
            json_path=args["json_path"],
            force=args.get("force", False),
            max_calib_batches=args.get("max_calib_batches"),
            sanity=args.get("sanity", 0),          # Caffe 下先忽略 sanity（见实现内说明）
            verbose=args.get("verbose", False),
            use_dla=args.get("use_dla", False),
            dla_core=args.get("dla_core", 0),
            allow_gpu_fallback=args.get("allow_gpu_fallback", True),
            fp16=args.get("fp16", True),
        )

    else:
        raise ValueError(f"Unknown format: {format!r}")
    
def _ensure_args(required_keys, args: dict):
    missing = [k for k in required_keys if k not in args]
    if missing:
        raise KeyError(f"Missing required args to build int8 engine: {missing}")
    

def build_int8_engine_caffe(prototxt, caffemodel, input_name, engine_path, npz_path,
                            shape, ws_mb, algo, cache_path, txt_path, json_path,
                            force=False, max_calib_batches=None, sanity=0, verbose=False,
                            use_dla=False, dla_core=0, allow_gpu_fallback=True, fp16=True):
    """
    从 Caffe (prototxt + caffemodel) 构建 INT8 引擎并导出 calib.{txt,json}
    - 复用 NPZ*Calibrator, export_txt_json_from_profile
    - DLA/FP16/GPU_FALLBACK 可从入口 args 控制（保持与 ONNX 版一致的体验）
    """
    # 能力检测（部分 Jetson 版本可能无 CaffeParser）
    if not hasattr(trt, "CaffeParser"):
        raise RuntimeError("Current TensorRT build does not provide trt.CaffeParser")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 0
    network = builder.create_network(flags)
    parser  = trt.CaffeParser()

    # 解析 caffe
    trt_model = parser.parse(prototxt, caffemodel, network, trt.float32)
    if trt_model is None:
        raise RuntimeError("Caffe parse failed. Check prototxt/caffemodel correctness.")

    vprint(verbose, f"[Build] Parsed CAFFE: {prototxt} / {caffemodel}")
    vprint(verbose, f"[Build] num_inputs={network.num_inputs}, num_outputs={network.num_outputs}, num_layers={network.num_layers}")

    # 统一输入：强制指定第 0 个输入的名称与静态 shape
    if network.num_inputs == 0:
        raise RuntimeError("No input tensors found in Caffe network (check your deploy).")
    inp = network.get_input(0)
    inp.name = input_name
    B, C, H, W = shape
    inp.shape = (C, H, W)
    print(f"[Build] input: {inp.name}  caffe-shape: {tuple(inp.shape)}  dtype: {str(inp.dtype)}")

    # === 自动标记输出 ===
    if network.num_outputs == 0:
        # 找最后一层
        last_layer = network.get_layer(network.num_layers - 1)
        if last_layer is None:
            raise RuntimeError("Caffe network has no layers, cannot mark output.")
        for i in range(last_layer.num_outputs):
            out_tensor = last_layer.get_output(i)
            if out_tensor is not None:
                network.mark_output(out_tensor)
                print(f"[Build] Auto-marked output: {out_tensor.name}")

    # 构建配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws_mb) << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if allow_gpu_fallback:
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # DLA（可选，注意静态 shape & 回退）
    if use_dla:
        runtime = trt.Runtime(logger)
        if runtime.num_dla_cores <= 0:
            print("[Warn] DLA requested but no DLA cores available on this platform. Fallback to GPU.")
        else:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(dla_core)

    # 选择校准器（与你 ONNX 版一致）
    Calib = NPZMinMaxCalibrator if algo == "minmax" else NPZEntropyCalibrator
    in_dtype_trt = inp.dtype
    in_dtype_np  = trt_dtype_to_np(in_dtype_trt)
    calib = Calib(npz_path, batch_size=B, cache_file=cache_path,
                  max_calib_batches=max_calib_batches, force=force,
                  in_dtype_np=in_dtype_np, verbose=verbose)
    config.int8_calibrator = calib
    print(f"[Build] Using calibrator: {Calib.__name__}")

    # 构建
    print("[Build] Building INT8 engine (CAFFE) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[OK] engine: {engine_path}")
    print(f"[OK] cache : {cache_path}")

    # 导出 scale / range
    try:
        scales = parse_trt_calib_cache_to_scales(cache_path, verbose=verbose)
    except Exception as e:
        print(f"[Warn] parse calib cache failed, fallback to scale=1.0: {e}")
        scales = None

    header = f"TRT-INT8-{('MinMax' if algo=='minmax' else 'EntropyCalibration2')}"
    export_txt_json_from_profile(network, header, txt_path, json_path, ranges_dict=scales)

    # Caffe 下的 sanity：你现有 sanity_compare 只支持 ONNX，这里先提示后跳过
    if sanity and sanity > 0:
        print("[Info] sanity requested, but current sanity_compare is ONNX-only. Skipped for CAFFE.")

    return engine_path


def build_int8_engine_onnx(onnx, engine_path, npz_path, shape, ws_mb, algo, cache_path,
                      txt_path, json_path, force=False, max_calib_batches=None, sanity=0, verbose=False):  # >>> CHANGED

    # define basic TRT components
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser  = trt.OnnxParser(network, logger)

    # parsing
    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: parse onnx failed")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(1)

    # Verbose ONNX / network info
    vprint(verbose, f"[Build] Parsed ONNX: {onnx}")
    vprint(verbose, f"[Build] num_inputs={network.num_inputs}, num_outputs={network.num_outputs}, num_layers={network.num_layers}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        vprint(verbose, f"[Build] Input[{i}]: name={inp.name} shape={tuple(inp.shape)} dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        vprint(verbose, f"[Build] Output[{i}]: name={out.name} shape={tuple(out.shape)} dtype={out.dtype}")

    # fetch first tensor as input
    inp = network.get_input(0)
    print(f"[Build] input: {inp.name}  onnx-shape: {tuple(inp.shape)}  dtype: {str(inp.dtype)}")

    # building config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws_mb) << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    vprint(verbose, f"[Build] workspace={ws_mb} MB  algo={algo}  INT8=on")

    # tensor profile for dynamic shape
    B,C,H,W = shape
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (B,C,H,W), (B,C,H,W), (B,C,H,W)) # fixed shape for simplicity
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)
    vprint(verbose, f"[Build] opt profile fixed to {(B,C,H,W)}")
    
    # verbose review: check all batches of inputs in network
    for i in range(network.num_inputs):
        ii = network.get_input(i)
        vprint(verbose, f"[Build] Input[{i}] name={ii.name} expect_shape={tuple(ii.shape)} expect_dtype={ii.dtype}")

    # define calibration and input dtype
    Calib = NPZMinMaxCalibrator if algo == "minmax" else NPZEntropyCalibrator
    in_dtype_trt = inp.dtype  # e.g., trt.DataType.HALF
    in_dtype_np  = trt_dtype_to_np(in_dtype_trt)
    if verbose:
        print(f"[Build] Input[0] name={inp.name} expect_shape={tuple(inp.shape)} expect_dtype={in_dtype_trt}")
        print(f"[Build] Upload will use numpy dtype: {in_dtype_np}")

    # create calibrator
    calib = Calib(npz_path, batch_size=B, cache_file=cache_path,
              max_calib_batches=max_calib_batches, force=force,
              in_dtype_np=in_dtype_np, verbose=verbose)
    config.int8_calibrator = calib
    print(f"[Build] Using calibrator: {Calib.__name__}")

    # execute calibration - get serialized engine & auto generated cache by trt
    print("[Build] Building INT8 engine ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")

    # serialized -> engine
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[OK] engine: {engine_path}")
    print(f"[OK] cache : {cache_path}")

    # extract scale numbers from cache
    try:
        scales = parse_trt_calib_cache_to_scales(cache_path, verbose=verbose)
    except Exception as e:
        print(f"[Warn] parse calib cache failed, fallback to scale=1.0: {e}")
        scales = None

    # write scale numbers into the required format
    header = f"TRT-INT8-{('MinMax' if algo=='minmax' else 'EntropyCalibration2')}"
    export_txt_json_from_profile(network, header, txt_path, json_path, ranges_dict=scales) 

    # optional sanity：build a FP32 baseline with small sample and compare the current int8 with it. 
    if sanity and sanity > 0:
        try:
            sanity_compare_fp32_vs_int8(onnx, engine_path, npz_path, B, C, H, W, sanity, verbose=verbose)
        except Exception as e:
            print(f"[Warn] sanity compare failed: {e}")


