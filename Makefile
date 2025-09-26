# ===== 可调参数（命令行可覆盖：make build TAG=dev-r35.4.1 BASE_IMAGE=...） =====
IMAGE       ?= trt_env
TAG         ?= dev
NAME        ?= trt_dev
PROJECT_DIR ?= $(HOME)/Documents/emass/pipe_trt_engine
BASE_IMAGE  ?= nvcr.io/nvidia/l4t-ml:r35.2.1-py3

UID         := $(shell id -u)
GID         := $(shell id -g)
VCS_REF     := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

BUILD_ARGS  := --build-arg BASE_IMAGE=$(BASE_IMAGE) \
               --build-arg HOST_UID=$(UID) \
               --build-arg HOST_GID=$(GID) \
               --build-arg VCS_REF=$(VCS_REF)

.PHONY: build rebuild up in down logs ps pull-base vars

# 若装了 pip-tools，则自动从 requirements.in 生成锁定文件
requirements.txt: requirements.in
	@if command -v pip-compile >/dev/null 2>&1; then \
		echo "[make] pip-compile requirements.in -> requirements.txt"; \
		pip-compile -o requirements.txt requirements.in; \
	else \
		echo "[make] pip-compile 未安装，沿用现有 requirements.txt（若不存在则创建空文件）"; \
		touch requirements.txt; \
	fi

build: docker/Dockerfile requirements.txt
	@echo "[make] building $(IMAGE):$(TAG) (BASE_IMAGE=$(BASE_IMAGE), VCS_REF=$(VCS_REF))"
	docker build -f docker/Dockerfile $(BUILD_ARGS) -t $(IMAGE):$(TAG) .

rebuild: docker/Dockerfile requirements.txt
	@echo "[make] rebuilding (no cache) $(IMAGE):$(TAG)"
	docker build --no-cache -f docker/Dockerfile $(BUILD_ARGS) -t $(IMAGE):$(TAG) .

pull-base:
	@echo "[make] pulling base image: $(BASE_IMAGE)"
	-docker pull $(BASE_IMAGE)

up:
	@docker ps --format '{{.Names}}' | grep -qx '$(NAME)' || \
	docker run -d \
	  --runtime nvidia \
	  -e NVIDIA_VISIBLE_DEVICES=all \
	  -e NVIDIA_DRIVER_CAPABILITIES=all \
	  -v $(PROJECT_DIR):/workspace:rw \
	  -w /workspace \
	  --user $(UID):$(GID) \
	  --name $(NAME) \
	  --label seakon.trt=1 \
	  $(IMAGE):$(TAG) \
	  sleep infinity
	@echo "[make] $(NAME) up."

in:
	@$(MAKE) -s up
	docker exec -it $(NAME) bash

down:
	-@docker rm -f $(NAME) >/dev/null 2>&1 && echo "[make] $(NAME) removed." || echo "[make] $(NAME) not running."

logs:
	docker logs -f $(NAME)

ps:
	docker ps --filter name=$(NAME)

vars:
	@echo IMAGE=$(IMAGE)
	@echo TAG=$(TAG)
	@echo NAME=$(NAME)
	@echo PROJECT_DIR=$(PROJECT_DIR)
	@echo BASE_IMAGE=$(BASE_IMAGE)
	@echo UID=$(UID) GID=$(GID) VCS_REF=$(VCS_REF)

