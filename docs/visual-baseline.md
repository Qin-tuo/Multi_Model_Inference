# 视觉基线复现入口

## B1 模型接入验证

2026-10-19：B1 尚未冻结，默认服务仍使用 B0 配置。按用户最新决定，本阶段深度直接采用 D435i 原生深度和 RealSense SDK 对齐，不再要求神经网络深度推理。RTMW3D-X 姿态和 YuNet 人脸均已生成 TensorRT 引擎并通过有限固定输入的数值核验。证据与未完成项见 [可部署性记录](../benchmarks/results/b1-feasibility-20261019.json)；真人场景精度、完整多模型链路和 B1 性能仍待验收。

原生深度复现入口使用应用的 `Camera` 类，检查时间对应、SDK 对齐、米制单位、无效点和关闭后的释放。输出目录每次使用新名称：

```bash
./05_runtime/environments/edge-assistant/bin/python scripts/probe_rgbd_camera.py \
  --output 05_runtime/validation/b1-sdk-rgbd-new
```

`vision.backend: realsense` 选择原生 RGBD，`v4l2` 保留 B0 的 RGB 路径。SDK 采集为 640×480、15 FPS，应用发布频率受 `capture_fps` 控制；这些配置值不是模型处理帧率。只有最新一组保留原始深度，历史环形缓存继续仅保存 JPEG。深度无效点保持 NaN 和独立掩码，不当作零距离；SDK 非零值仍需结合工作距离、目标区域和场景样本验证。依赖版本见 [运行依赖](../01_src/deploy/vision/requirements-runtime.lock)，不升级系统 BSP/CUDA/TensorRT。

模型来源、校验值与预处理在 [候选模型清单](../03_models/candidates/vision-b1.json)。以人脸为例：

```bash
./05_runtime/environments/edge-assistant/bin/python scripts/prepare_vision_b1.py --role face
./05_runtime/environments/edge-assistant/bin/python scripts/build_vision_b1.py \
  --role face --workspace-mib 512 --output 05_runtime/validation/b1-face-new
./05_runtime/environments/edge-assistant/bin/python scripts/probe_vision_b1.py \
  --role face --build-root 05_runtime/validation/b1-face-new \
  --output 05_runtime/validation/b1-face-new/parity
```

构建和核验会独占本项目的构建锁，暂时停止本项目服务，并在结束或失败后恢复原先运行的组件。当前模型验证采用 FP32、关闭 TF32；编译 workspace 上限不等于运行时内存。CPU ONNX Runtime 和 OpenCV FaceDetectorYN 仅作为独立核验参考，正式视觉神经网络使用 TensorRT。YuNet 的检测框及五点解码另与 OpenCV 官方实现比较，覆盖有人脸、镜像和黑色负样本。

相机元数据探针使用独立工具环境，不启动视频流：

```bash
./05_runtime/builds/vision-tools/bin/python scripts/probe_stereo_metadata.py \
  --output 05_runtime/validation/b1-camera-new/profiles.json
```

工具环境版本在 [requirements-tools.lock](../01_src/deploy/vision/requirements-tools.lock)。已完成 60 组四路采集检查及应用内 30 帧 RGBD 检查；短窗口不等于长期稳定性或绝对测距精度。IR 和 RGB 内参、外参不同，正式应用直接使用 `rs.align(stream.color)` 的结果，并保留每路原始时间戳。相近 SDK 全局时间不代表硬件同时曝光。

神经双目深度已移至后续对照实验，代码在 `01_src/benchmarks/stereo_depth_experiment.py` 与 `stereo_geometry.py`。已有引擎和失败的红外域比较记录保留：转换一致性通过，但真实样本相对 SDK 的尾部偏差较大；SDK 本身也不是地面真值。该实验不参与当前 B1 完成判定，准备／构建脚本默认跳过它，只有显式 `--role depth` 才会运行。姿态模型的相对 z 始终不冒充相机测距。

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
