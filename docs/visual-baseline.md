# 视觉基线复现入口

## B1 模型接入验证

2026-10-19：B1 尚未冻结，当前默认程序仍使用 B0 模型。RTMW3D-X 姿态和 C-Fast-FoundationStereo 深度均已在本板生成 TensorRT 引擎，各通过两组固定输入的 ONNX/TensorRT 数值核验。证据与未完成项见 [可部署性记录](../benchmarks/results/b1-feasibility-20261019.json)；这不代表真人场景精度、实时链路或整机性能已经验收。

模型来源、校验值与预处理在 [候选模型清单](../03_models/candidates/vision-b1.json)。从项目根目录复现深度验证，输出目录每次使用新名称：

```bash
./05_runtime/environments/edge-assistant/bin/python scripts/prepare_vision_b1.py --role depth
./05_runtime/environments/edge-assistant/bin/python scripts/build_vision_b1.py \
  --role depth --workspace-mib 4096 --output 05_runtime/validation/b1-depth-new
./05_runtime/environments/edge-assistant/bin/python scripts/probe_vision_b1.py \
  --role depth --build-root 05_runtime/validation/b1-depth-new \
  --output 05_runtime/validation/b1-depth-new/parity
```

构建和核验会独占本项目的构建锁，暂时停止本项目服务，并在结束或失败后恢复原先运行的组件。当前验证采用 FP32、关闭 TF32；4096 MiB 是编译 workspace 上限，运行时内存需另行测量。核验中的 CPU ONNX Runtime 仅用于生成参考输出，正式视觉推理使用 TensorRT。固定双目样本来自 `05_runtime/frameworks/b1/fast-foundationstereo/demo_data/`，提交版本与输入哈希记录在核验报告中。

相机元数据探针使用独立工具环境，不启动视频流：

```bash
./05_runtime/builds/vision-tools/bin/python scripts/probe_stereo_metadata.py \
  --output 05_runtime/validation/b1-camera-new/profiles.json
```

工具环境版本在 [requirements-tools.lock](../01_src/deploy/vision/requirements-tools.lock)。已读到 D435i 的左右 IR、RGB 与 SDK 深度参考配置及约 49.83 mm 双目基线；仍需验证同时采集、时间对应与真实测距。IR 和 RGB 的内参、外参不同，不能直接用 RGB 检测框坐标索引 IR 深度。SDK 深度仅作为对照来源。

`01_src/app/stereo_geometry.py` 按同一像素坐标下的视差、焦距和基线换算米制深度，并保留无效标记。有限正视差仅证明数值有效，遮挡、错匹配和传感器域适配还需单独验证；姿态模型的相对 z 不参与该换算。

## 已冻结的 B0

先从项目根目录启动完整程序，确认相机与四个服务正常，再理解各模块职责。当前名称为“多模型端侧助手”，服务前缀为 `edge-assistant`。

```bash
./run.sh services up
./run.sh services status
```

下面两个脚本分别测量 V1 单视觉模块和 V2 连续视觉链路。它们会临时停止或切换本项目服务，结束后恢复原配置；每次使用新的输出目录，测量期间不要同时运行其他测试、构建或下载任务。

```bash
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_modules.py \
  --output 05_runtime/validation/vision-v1-new
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_stream.py \
  --output 05_runtime/validation/vision-v2-new
```

固定输入在 `05_runtime/datasets/b0-sw-v1/`。脚本校验 [B0 协议](../benchmarks/manifests/b0-sw-v1.json)、原始配置、模型及输入哈希，路径兼容层将历史目录映射到当前目录；冻结记录保持原字节。

对照 [原始视觉汇总](../benchmarks/results/b0-vision-summary.json) 阅读结果，关注分模块 P50/P95、感知完成率、结果帧龄、无结果比例与资源占用。当前相机配置 10 FPS、感知配置 5 FPS；不要将配置值当作最大吞吐，也不要把网页刷新速度当作板端推理速度。后续由你亲自复现、提出假设，再逐项分析和优化。
