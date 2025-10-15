# ----------- Config ---------------
include nvdla_vp.config
# MODEL_NAME=lenet
# NVDLA_VER=nv_small
# VP_IMG_NAME=nvdla-vp
# INFORMAT=nchw
# SC_LOG_LVL=sc_debug
# PROFILE=fast-math (basic|default|performance|fast-math)


# -------- VP settings --------
UID := $(shell id -u)
GID := $(shell id -g)




# # 镜像
# VP_IMG          := $(VP_IMG_NAME):$(NVDLA_VER)
# VP_IMG_VER      := --build-arg PROJECT_NAME=$(NVDLA_VER)


# PY_IMG          := py-standard

# # 宿主工程目录 → 容器内 /usr/local/nvdla/host
# WORK_DIR        ?= /usr/local/nvdla
# HOST_MOUNT      ?= $(WORK_DIR)/host

# # 本机工程绝对路径（避免 $PWD 在某些 shell 不展开）
# MAKEFILE_SELF   := $(abspath $(firstword $(MAKEFILE_LIST)))
# MK_DIR          := $(abspath $(dir $(MAKEFILE_SELF)))
# HOST_ABS        := $(MK_DIR)

# # 先假设镜像里文件在 vp/ 目录；如果你确认不是，改这里
# VP_CONF         ?= $(WORK_DIR)/aarch64_nvdla.lua
# VP_EXEC         ?= $(WORK_DIR)/vp/build/build/bin/aarch64_toplevel
# # VP_EXEC         ?= /usr/bin/aarch64_toplevel

# # 模型配置
# ONNX            ?= $(MK_DIR)/models/$(MODEL_NAME).onnx
# PROTOTXT        ?= $(MK_DIR)/models/$(MODEL_NAME).prototxt
# CAFFEMODEL      ?= $(MK_DIR)/models/$(MODEL_NAME).caffemodel
# OUT_NVDLA       ?= $(MK_DIR)/out/$(MODEL_NAME).nvdla
# VP_LOG_DIR      ?= $(MK_DIR)/logs
# VP_LOG          ?= $(VP_LOG_DIR)/$(MODEL_NAME)_vp.log

# # -------- compile ---------
# CALIB_JSON      ?= $(MODEL_NAME)_calib.json
# CALIB_DIR       ?= $(MK_DIR)/models/$(CALIB_JSON)
# CALIB_REL       := $(HOST_MOUNT)/models/$(CALIB_JSON)
# C_PRECISION     := $(if $(filter $(NVDLA_VER),nv_small),int8,fp16)
# CALIBRATION     := $(if $(filter $(NVDLA_VER),nv_small),--calibtable $(CALIB_REL),)

# # scripts
# PROTO2CAFFE_SCRIPT := $(MK_DIR)/scripts/prototxt2caffemodel.py

# OUT_DIR     ?= $(MK_DIR)/out
# LOG_DIR     ?= $(MK_DIR)/logs
# SW_DIR      ?= /opt/sw
# # COMPILER    ?= $(SW_DIR)/prebuilt/x86-ubuntu/nvdla_compiler
# COMPILER    ?= $(WORK_DIR)/nvdla_compiler

# REL_ONNX    := $(patsubst $(MK_DIR)/%,%,$(ONNX))
# REL_PROTO   := $(patsubst $(MK_DIR)/%,%,$(PROTOTXT))
# REL_CAFFE   := $(patsubst $(MK_DIR)/%,%,$(CAFFEMODEL))
# REL_OUT     := $(patsubst $(MK_DIR)/%,%,$(OUT_DIR))


# DOCKERFILE  ?= $(MK_DIR)/docker/Dockerfile.$(VP_IMG_NAME)

# MODELS_DIR      ?= $(MK_DIR)/models
# DATA_DIR        ?= $(MK_DIR)/data
# SCRIPT_DIR      ?= $(MK_DIR)/scripts

# REL_MODELS    := $(patsubst $(MK_DIR)/%,%,$(MODELS_DIR))
# REL_DATA      := $(patsubst $(MK_DIR)/%,%,$(DATA_DIR))
# REL_SCRIPT      := $(patsubst $(MK_DIR)/%,%,$(SCRIPT_DIR))

# IMG_OUT_ROOT    ?= $(DATA_DIR)/img
# IMG_COUNT       ?= 1
# NPZ_FILE        := $(MODELS_DIR)/$(MODEL_NAME).npz
# NPZ2IMG         := $(MK_DIR)/scripts/npz2img.py








#### 基本定位（不要改） ####
MAKEFILE_SELF   := $(abspath $(firstword $(MAKEFILE_LIST)))
HOST_ROOT       := $(abspath $(dir $(MAKEFILE_SELF)))

# WORK_DIR        ?= /usr/local/nvdla
# HOST_MOUNT      ?= $(WORK_DIR)/host
# MK_DIR          := $(abspath $(dir $(MAKEFILE_SELF)))

#### 容器内根路径（通常用 /usr/local/nvdla） ####
CONT_ROOT       ?= /usr/local/nvdla
CONT_HOST       ?= $(CONT_ROOT)/host


ONNX2PROTO_IMG  := onnx2prototxt
PROTO2CAFFE_IMG := bvlc/caffe:cpu
VP_IMG          := $(VP_IMG_NAME):$(NVDLA_VER)
VP_IMG_VER      := --build-arg NVDLA_VER=$(NVDLA_VER)
DOCKERFILE      := $(HOST_ROOT)/docker/Dockerfile.$(VP_IMG_NAME)

#### 工程内的相对目录（全是相对 HOST_ROOT 的路径名）####
MODELS_DIR   ?= models
DATA_DIR     ?= data
SCRIPTS_DIR  ?= scripts
OUT_DIR      ?= out
LOG_DIR      ?= logs

#### 以“宿主绝对路径”引用（用于宿主文件检查、写入产物）####
H_MODELS := $(HOST_ROOT)/$(MODELS_DIR)
H_DATA   := $(HOST_ROOT)/$(DATA_DIR)
H_SCRIPTS:= $(HOST_ROOT)/$(SCRIPTS_DIR)
H_OUT    := $(HOST_ROOT)/$(OUT_DIR)
H_LOG    := $(HOST_ROOT)/$(LOG_DIR)

#### 以“容器可见路径”引用（容器命令里用 CONT_HOST 前缀）####
C_MODELS := $(CONT_HOST)/$(MODELS_DIR)
C_DATA   := $(CONT_HOST)/$(DATA_DIR)
C_SCRIPTS:= $(CONT_HOST)/$(SCRIPTS_DIR)
C_OUT    := $(CONT_HOST)/$(OUT_DIR)
C_LOG    := $(CONT_HOST)/$(LOG_DIR)

#### 模型名（不带后缀）####
MODEL_NAME ?= lenet

#### 相对文件名（相对工程根）####
REL_ONNX      := $(MODELS_DIR)/$(MODEL_NAME).onnx
REL_PROTOTXT  := $(MODELS_DIR)/$(MODEL_NAME).prototxt
REL_CAFFE     := $(MODELS_DIR)/$(MODEL_NAME).caffemodel
REL_NVDLA_OUT := $(OUT_DIR)/$(MODEL_NAME).nvdla

#### 宿主路径（检查/产物落地用）####
ONNX        := $(H_MODELS)/$(MODEL_NAME).onnx
PROTOTXT    := $(H_MODELS)/$(MODEL_NAME).prototxt
CAFFEMODEL  := $(H_MODELS)/$(MODEL_NAME).caffemodel
OUT_NVDLA   := $(H_OUT)/$(MODEL_NAME).nvdla
VP_LOG      := $(H_LOG)/$(MODEL_NAME)_vp.log
SC_LOG      ?= $(H_LOG)/sc-$(MODEL_NAME).log

#### 容器可见路径（传给容器里的工具）####
C_REL_ONNX      := $(C_MODELS)/$(MODEL_NAME).onnx
C_REL_PROTOTXT  := $(C_MODELS)/$(MODEL_NAME).prototxt
C_REL_CAFFE     := $(C_MODELS)/$(MODEL_NAME).caffemodel
C_REL_OUTDIR    := $(C_OUT)

#### 校准（small 用 INT8 + 校准表；full 默认 FP16）####
CALIB_JSON   ?= $(MODEL_NAME)_calib.json
REL_CALIB    := $(MODELS_DIR)/$(CALIB_JSON)
H_CALIB      := $(H_MODELS)/$(CALIB_JSON)
C_CALIB      := $(C_MODELS)/$(CALIB_JSON)

C_PRECISION  := $(if $(filter $(NVDLA_VER),nv_small),int8,fp16)
CALIBRATION  := $(if $(filter $(NVDLA_VER),nv_small),--calibtable $(C_CALIB),)
INFORMAT_ARG := $(if $(strip $(INFORMAT)),--informat $(INFORMAT),)


#### 容器内工作根&工具 ####
WORK_DIR      ?= $(CONT_ROOT)
COMPILER      ?= $(CONT_ROOT)/nvdla_compiler
SW_PREBUILT   ?= /opt/sw/prebuilt/x86-ubuntu

#### VP 可执行（按你的镜像安装路径来；示例保留两种常见路径）####
VP_EXEC    ?= $(CONT_ROOT)/vp/build/build/bin/aarch64_toplevel
# 若你安装到 vp/build/bin，也可以用：
# VP_EXEC  ?= $(CONT_ROOT)/vp/build/bin/aarch64_toplevel

#### VP Lua 配置（通常放在 WORK_DIR 下）####
VP_CONF    ?= $(CONT_ROOT)/aarch64_nvdla.lua

SC_ENV     := SC_SIGNAL_WRITE_CHECK=DISABLE \
              SC_LOG="outfile:$(SC_LOG);verbosity_level:$(SC_LOG_LVL);csb_adaptor:enable;dbb_adaptor:enable"


# SW_DIR      ?= /opt/sw
# # COMPILER    ?= $(SW_DIR)/prebuilt/x86-ubuntu/nvdla_compiler
# COMPILER    ?= $(WORK_DIR)/nvdla_compiler

.PHONY: vp-compile vp-run vp-shell build-onnx2prototxt build-proto2caffe vp-build vp-compiler-compile

# 在容器里调用镜像自带的 nvdla_compiler 编译 Caffe 模型为 .nvdla
vp-run:
	docker run --rm -it \
	  -v $(HOST_ROOT):$(CONT_HOST) \
	  -w $(CONT_ROOT) \
	  $(VP_IMG) \
	  /bin/bash -lc 'echo "[DEBUG] SC_LOG inside container: $$SC_LOG"; exec $(VP_EXEC) -c $(VP_CONF)'

# 启动交互 shell（排障/探索用）
vp-shell:
	docker run --rm -it \
	  -v $(HOST_ROOT):$(CONT_HOST) \
	  -w $(CONT_ROOT) \
	  $(VP_IMG) \
	  /bin/bash


# 从宿主快速筛出 CSB/DBB 关键行
vp-log:
	@grep -E 'nvdla\.csb_adaptor|nvdla\.dbb_adaptor' $(VP_LOG) | head || true

vp-clean:
	rm -rf nvdla_vp/out/* nvdla_vp/logs/*

# ========= 新增：在 VP 容器里 clone sw 并用 prebuilt 的编译器生成 .nvdla =========
# 用法： make vp-compile-sw MODEL_NAME=my_model
# 依赖：models/$(MODEL_NAME).prototxt + models/$(MODEL_NAME).caffemodel


vp-build:
	@echo "[INFO] Building $(VP_IMG) with $(DOCKERFILE)"
	docker build \
	  -t $(VP_IMG) \
	  -f $(DOCKERFILE) \
	  --build-arg NVDLA_VER=$(NVDLA_VER) \
	  --build-arg ROOT_DIR=$(CONT_ROOT) \
	  --build-arg SC_LOG_DIR=$(CONT_ROOT)/host/logs/sc.log \
	  --build-arg SC_LOG_LVL=$(SC_LOG_LVL) \
	  --no-cache \
	  $(HOST_ROOT)


vp-compile:
	@if ! docker image inspect $(VP_IMG) >/dev/null 2>&1; then \
		echo "[INFO] Image $(VP_IMG) not found, building..."; \
		$(MAKE) -C $(HOST_ROOT) -f $(MAKEFILE_SELF) vp-build; \
	else \
		echo "[INFO] Image $(VP_IMG) found."; \
	fi
	@mkdir -p $(OUT_DIR)
	@echo "[i] Clean: $(OUT_DIR)"
	@docker run --rm \
	  -v $(HOST_ROOT):$(CONT_HOST) \
	  $(VP_IMG) \
	  /bin/bash -lc '\
	    rm -rf \
	      $(C_OUT)/wisdom.dir \
	      $(C_OUT)/*.nvdla \
	      $(C_OUT)/*.protobuf \
	      $(C_OUT)/*.log || true; \
	    mkdir -p $(C_OUT) \
	  '
	@echo "[i] Using compiler: $(COMPILER) (image: $(VP_IMG))"
	@test -f "$(PROTOTXT)"   || (echo "[ERR] Prototxt not found: $(PROTOTXT)"; exit 1)
	@test -f "$(CAFFEMODEL)" || (echo "[ERR] Caffemodel not found: $(CAFFEMODEL)"; exit 1)
	@echo "[i] Compiling to wisdom dir: $(C_OUT)"
	@docker run --rm \
	  -v $(HOST_ROOT):$(CONT_HOST) \
	  -w $(SW_PREBUILT) \
	  -e LD_LIBRARY_PATH=$(CONT_ROOT) \
	  $(VP_IMG) \
	  /bin/bash -lc '\
	    set -e; \
	    $(COMPILER) \
	      --prototxt   $(C_REL_PROTOTXT) \
	      --caffemodel $(C_REL_CAFFE) \
	      --configtarget $(NVDLA_VER) \
	      --cprecision $(C_PRECISION) \
	      $(INFORMAT_ARG) \
		  $(CALIBRATION) \
	      --profile $(PROFILE) \
	      -o $(C_OUT); \
	    echo "[i] Checking/renaming output in $(C_OUT)"; \
		if [ -f "$(PROFILE).nvdla" ]; then \
			mv -v "$(PROFILE).nvdla" "$(C_OUT)/$(MODEL_NAME).nvdla"; \
		else \
			echo "[WARN] $(C_OUT)/$(PROFILE).nvdla not found (compiler may have used a different name)"; \
			ls -lh "$(C_OUT)"; \
		fi \
	  ' 2>&1 | tee $(LOG_DIR)/compile.log
	@echo "[i] Output listing:"
	@ls -lh $(OUT_DIR)


vp-img-data:
	@mkdir -p $(IMG_OUT_ROOT) $(LOG_DIR)
	@test -f "$(NPZ_FILE)" || (echo "[ERR] NPZ not found: $(NPZ_FILE)"; exit 1)
	@echo "[i] NPZ: $(NPZ_FILE)"
	@echo "[i] Out: $(IMG_OUT_ROOT)/$(MODEL_NAME)   Count: $(IMG_COUNT)"
	@docker run --rm \
	  -v $(HOST_ROOT):/work \
	  -w /work \
	  --entrypoint bash \
	  $(ONNX2PROTO_IMG) -lc '\
	    pip install --no-cache-dir numpy pillow && \
	    python3 $(REL_SCRIPT)/npz2img.py \
			--npz $(REL_MODELS)/$(MODEL_NAME).npz \
			--count $(IMG_COUNT) \
			--out-root $(REL_DATA)/img \
	  '
	@echo "[i] Sample listing:"
	@ls -lh $(IMG_OUT_ROOT)/$(MODEL_NAME) | head


build-onnx2proto:
	@echo "[i] Building $(ONNX2PROTO_IMG)"
	docker build -t $(ONNX2PROTO_IMG) --no-cache -f $(HOST_ROOT)/docker/Dockerfile.$(ONNX2PROTO_IMG) $(HOST_ROOT)
	@echo "[i] Converting ONNX -> prototxt"
	@test -f "$(ONNX)" || (echo "[ERR] ONNX not found: $(ONNX)"; exit 1)
	docker run --rm -v $(HOST_ROOT):/work $(ONNX2PROTO_IMG) \
	  /work/$(REL_ONNX) --out /work/$(REL_PROTO)
	@echo "[OK] Wrote: $(PROTOTXT)"

build-proto2caffe:
	@echo "[i] Generating Caffe model (.caffemodel) with $(PROTO2CAFFE_IMG)"
	@test -f "$(PROTOTXT)" || (echo "[ERR] Prototxt not found: $(PROTOTXT)"; exit 1)
	docker run --rm -v $(HOST_ROOT):/work -w /work $(PROTO2CAFFE_IMG) \
	  bash -lc 'python /work/$(patsubst $(HOST_ROOT)/%,%,$(PROTO2CAFFE_SCRIPT)) \
	    /work/$(REL_PROTO) \
	    /work/$(REL_CAFFE)'
	@echo "[OK] Wrote: $(CAFFEMODEL)"

