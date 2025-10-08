# -------- VP settings --------
UID := $(shell id -u)
GID := $(shell id -g)

# 使用官方镜像
VP_IMG        ?= nvdla/vp

# 宿主工程目录 → 容器内 /usr/local/nvdla/host
HOST_MOUNT    ?= /usr/local/nvdla/host

# 本机工程绝对路径（避免 $PWD 在某些 shell 不展开）
HOST_ABS      := $(shell pwd)

# 先假设镜像里文件在 vp/ 目录；如果你确认不是，改这里
VP_CONF       ?= /usr/local/nvdla/aarch64_nvdla.lua

# 模型配置
MODEL_NAME    ?= lenet
PROTOTXT      ?= nvdla_vp/models/$(MODEL_NAME).prototxt
CAFFEMODEL    ?= nvdla_vp/models/$(MODEL_NAME).caffemodel
OUT_NVDLA     ?= nvdla_vp/out/$(MODEL_NAME).nvdla
VP_LOG_DIR    ?= nvdla_vp/logs
VP_LOG        ?= $(VP_LOG_DIR)/$(MODEL_NAME)_vp.log

.PHONY: vp-pull vp-compile vp-run vp-shell vp-log vp-clean

vp-pull:
	@docker pull $(VP_IMG)

# 在容器里调用镜像自带的 nvdla_compiler
vp-compile:
	@mkdir -p nvdla_vp/out
	docker run --rm \
	  -v $(HOST_ABS):$(HOST_MOUNT) \
	  $(VP_IMG) \
	  /usr/local/nvdla/nvdla_compiler \
	    --prototxt   $(HOST_MOUNT)/$(PROTOTXT) \
	    --caffemodel $(HOST_MOUNT)/$(CAFFEMODEL) \
	    --output     $(HOST_MOUNT)/$(OUT_NVDLA)

vp-run:
	@mkdir -p $(VP_LOG_DIR)
	docker run --rm -it \
	  -v $(HOST_ABS):$(HOST_MOUNT) \
	  -w /usr/local/nvdla \
	  $(VP_IMG) \
	  /usr/bin/aarch64_toplevel -c $(VP_CONF) \
	  2>&1 | tee $(VP_LOG)

vp-shell:
	docker run --rm -it \
	  -v $(HOST_ABS):$(HOST_MOUNT) \
	  -w /usr/local/nvdla \
	  $(VP_IMG) \
	  /bin/bash


# 从宿主快速筛出 CSB/DBB 关键行
vp-log:
	@grep -E 'nvdla\.csb_adaptor|nvdla\.dbb_adaptor' $(VP_LOG) | head || true

vp-clean:
	rm -rf nvdla_vp/out/* nvdla_vp/logs/*
