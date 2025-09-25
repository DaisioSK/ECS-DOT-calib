每个目录的用途（先记概念）

src/：项目代码（app=入口/CLI，core=TRT封装，io=数据I/O与预处理，metrics=评估，utils=通用工具）。

scripts/：命令行脚本（如 build_engine.py / run_infer.py / bench.sh）。

configs/：YAML 配置（model/ 定义模型名与shape，runtime/ 定义 GPU/DLA/精度/批量等运行参数）。

models/：导出的 ONNX、构建好的 TensorRT 引擎、INT8 校准缓存。

data/：数据存放（raw/ 原始，processed/ 处理后，samples/ 小样本示例）。

tests/：pytest 单元/集成测试。

reports/：评测/基准的 JSON/CSV 输出。

docker/：Dockerfile 与相关脚本（后续再写）。

.vscode/：工作区设置与调试配置（后续再放 settings/launch）。
