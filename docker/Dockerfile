# ========= 1) 固定基底（不要 latest），日后升级只改这行或改 digest =========
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r35.2.1-py3
FROM ${BASE_IMAGE}

# ========= 2) 元信息（可注入 git 短 SHA） =========
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="trt_env_dev" \
      org.opencontainers.image.description="TRT dev runtime on top of ${BASE_IMAGE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      maintainer="you <you@example.com>"

# ========= 3) 基本环境变量（减少 apt 交互、统一时区/编码、静音 pip 提示） =========
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Singapore \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV PATH=/usr/src/tensorrt/bin:${PATH}

# ========= 4) 系统依赖（按需精简；最后清理 apt 缓存减镜像体积） =========
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      vim less nano curl ca-certificates git \
      pkg-config build-essential \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ========= 5) 工作目录与 alias（系统级 bashrc，所有交互 shell 生效） =========
WORKDIR /workspace
RUN printf "alias ll='ls -alF'\nalias la='ls -A'\nalias l='ls -CF'\nalias py='python'\n" >> /etc/bash.bashrc

# ========= 6) 修复 “I have no name!”（为宿主 UID/GID 建账户，家目录设为 /workspace） =========
# 构建时传入宿主的 UID/GID：--build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g)
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN if ! getent group ${HOST_GID} >/dev/null; then groupadd -g ${HOST_GID} hostgroup; fi && \
    if ! getent passwd ${HOST_UID} >/dev/null; then \
      useradd -u ${HOST_UID} -g ${HOST_GID} -d /workspace -s /bin/bash hostuser; fi

# ========= 7) Python 依赖安装（空 requirements.txt 也不会报错） =========
# 修复 python 默认指向 2.7 的问题
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# 只拷贝锁定文件，提高缓存命中率；你用 pip-tools 生成的 requirements.txt 放仓库根
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -U pip wheel setuptools && \
    if grep -q '[^[:space:]]' /tmp/requirements.txt; then \
      python3 -m pip install -r /tmp/requirements.txt; \
    else \
      echo "requirements.txt is empty; skip pip install"; \
    fi && \
    rm -rf /root/.cache/pip

# ========= 8) 默认入口 =========
CMD ["bash"]

