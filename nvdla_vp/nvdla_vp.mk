# -------- VP settings --------
UID := $(shell id -u)
GID := $(shell id -g)

# 使用官方镜像
VP_IMG        ?= nvdla/vp

# 宿主工程目录 → 容器内 /usr/local/nvdla/host
WORK_DIR      ?= /usr/local/nvdla
HOST_MOUNT    ?= $(WORK_DIR)/host

# 本机工程绝对路径（避免 $PWD 在某些 shell 不展开）
MK_DIR        := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
HOST_ABS      := $(MK_DIR)

# 先假设镜像里文件在 vp/ 目录；如果你确认不是，改这里
VP_CONF       ?= /usr/local/nvdla/aarch64_nvdla.lua
VP_EXEC       ?= /usr/bin/aarch64_toplevel

# 模型配置
MODEL_NAME    ?= cifar_simplecnn
PROTOTXT      ?= models/$(MODEL_NAME).prototxt
CAFFEMODEL    ?= models/$(MODEL_NAME).caffemodel
OUT_NVDLA     ?= out/$(MODEL_NAME).nvdla
VP_LOG_DIR    ?= nvdla_vp/logs
VP_LOG        ?= $(VP_LOG_DIR)/$(MODEL_NAME)_vp.log
NVDLA_PROFILE ?= nv_small
C_PRECISION   ?= fp16  #（FP16，避免量化/校准）

.PHONY: vp-pull vp-compile vp-run vp-shell vp-log vp-clean

# 快速查看容器内目录结构（只读探索）
vp-inspect:
	docker run --rm -it \
	  -v $(HOST_ABS):$(HOST_MOUNT) \
	  -w /usr/local/nvdla \
	  $(VP_IMG) \
	  bash -lc '\
	    echo "== /usr/local/nvdla ==" && ls -al /usr/local/nvdla && \
	    echo "\n== tree /usr/local/nvdla (top 2 levels) ==" && \
	    (command -v tree >/dev/null || apt-get update && apt-get install -y tree >/dev/null 2>&1 || true) && \
	    tree -L 2 /usr/local/nvdla || true \
	  '
	  
vp-pull:
	@docker pull $(VP_IMG)

# 在容器里调用镜像自带的 nvdla_compiler 编译 Caffe 模型为 .nvdla
vp-compile:
	@mkdir -p out
	docker run --rm \
	  -e LD_LIBRARY_PATH=/$(WORK_DIR) \
	  -v $(HOST_ABS):$(HOST_MOUNT) \
	  $(VP_IMG) \
	  /usr/local/nvdla/nvdla_compiler \
	    --prototxt   $(HOST_MOUNT)/$(PROTOTXT) \
	    --caffemodel $(HOST_MOUNT)/$(CAFFEMODEL) \
	    -o $(HOST_MOUNT)/out \
        --configtarget $(NVDLA_PROFILE) \
        --cprecision $(C_PRECISION) \
        --profile    fast-math

# 启动 VP 并抓取 SystemC/驱动层 log
vp-run:
	@mkdir -p $(VP_LOG_DIR)
	docker run --rm -it \
	  -v $(HOST_ABS):$(HOST_MOUNT) \
	  -w $(WORK_DIR) \
	  $(VP_IMG) \
	  $(VP_EXEC) -c $(VP_CONF) \
	  2>&1 | tee $(VP_LOG)

# 启动交互 shell（排障/探索用）
vp-shell:
	docker run --rm -it \
	  -v $(HOST_ABS):$(HOST_MOUNT) \
	  -w $(WORK_DIR) \
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
vp-compile-sw:
	@mkdir -p out
	@echo "==> 编译: $(PROTOTXT) + $(CAFFEMODEL) -> $(OUT_NVDLA)"
	docker run --rm -it \
	  -v $(HOST_ABS):$(HOST_MOUNT) \
	  -w /root \
	  $(VP_IMG) \
	  bash -lc '\
	    set -e; \
	    echo "[prep] clone sw (若已存在则跳过)"; \
	    test -d sw || git clone --depth 1 https://github.com/nvdla/sw.git; \
	    cd sw/prebuilt/x86-ubuntu; \
	    echo "[compile] ./nvdla_compiler --prototxt $(HOST_MOUNT)/$(PROTOTXT) --caffemodel $(HOST_MOUNT)/$(CAFFEMODEL) --configtarget $(NVDLA_PROFILE) -o $(HOST_MOUNT)/out"; \
	    ./nvdla_compiler \
	      --prototxt   $(HOST_MOUNT)/$(PROTOTXT) \
	      --caffemodel $(HOST_MOUNT)/$(CAFFEMODEL) \
	      --configtarget $(NVDLA_PROFILE) \
	      --profile fast-math \
	      -o $(HOST_MOUNT)/out; \
	    echo "[done] 输出:"; ls -lh $(HOST_MOUNT)/out \
	  '
	  

# ========= 新增：打印 VP 内部要执行的命令（你复制进 VP 里粘贴即可） =========
vp-runtime-help:
	@echo "---------- 在 VP (Buildroot) 里运行以下命令 ----------"
	@echo "mount -t 9p -o trans=virtio,version=9p2000.L r /mnt"
	@echo "/usr/bin/nvdla_runtime --loadable /mnt/$(OUT_NVDLA) --rawdump | tee /mnt/$(VP_LOG)"
	@echo "------------------------------------------------------"