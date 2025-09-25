# TRT Pipeline (Jetson)

目标：在 Jetson 上以 Docker 为唯一运行环境，构建可复现的 TensorRT 推理流水线（支持 GPU/DLA），并以脚本 + 配置驱动，持续评测与优化。

## 仓库结构
- `src/`      核心代码（app/core/io/metrics/utils）
- `scripts/`  命令脚本（build_engine / run_infer / bench 等）
- `configs/`  YAML 配置（model 与 runtime）
- `models/`   onnx / engines / calib 产物（不进 Git）
- `data/`     数据（raw/processed/samples）
- `tests/`    单元/集成测试
- `reports/`  评测/基准产物（JSON/CSV）
- `docker/`   Dockerfile 等
- `.vscode/`  VS Code 工作区配置

## 快速开始（后续补充）
1) 准备 Docker 镜像并常驻容器
2) 进入容器运行脚本
3) 使用 configs 选择 GPU/DLA/精度参数
