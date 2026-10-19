# 视觉基线复现入口

## B1 模型接入验证

2026-10-19：B1 尚未冻结，默认服务仍使用 B0 配置。按用户最新决定，本阶段深度直接采用 D435i 原生深度和 RealSense SDK 对齐，不再要求神经网络深度推理。9 类所选视觉神经模型及 1 个视线辅助模型已完成有限固定输入的 TensorRT 核验。证据与未完成项见 [可部署性记录](../benchmarks/results/b1-feasibility-20261019.json)；RTMW3D 原始权重／导出链核对、注册库匹配、真人场景精度、完整多模型链路和 B1 性能仍待验收。

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

人脸特征和表情复用同一份人脸框及五点来源，各自遵循上游的 112×112 对齐方法。SFace 输出 128 维特征，匹配前做 L2 归一化；表情输出七类分值，包含 `happy`，不提供原始私有模型的 `exaggeration`。复现时先完成上面的人脸检测核验：

```bash
./05_runtime/environments/edge-assistant/bin/python scripts/build_vision_b1.py \
  --role face_identity expression --output 05_runtime/validation/b1-face-tasks-new
./05_runtime/environments/edge-assistant/bin/python scripts/probe_face_tasks_b1.py \
  --build-root 05_runtime/validation/b1-face-tasks-new \
  --face-build-root 05_runtime/validation/b1-face-new \
  --output 05_runtime/validation/b1-face-tasks-new/parity
```

检测分割复用官方 YOLOv8n-seg 权重，输出 80 类检测与原图大小的实例 mask。当前帧最多输出 10 个目标，输入像素和 mask 分配均有上限。已通过原始 Ultralytics PyTorch 的框、类别、置信度和 mask 对照，覆盖普通人物、多人公交场景、奇数尺寸及空白负样本。

```bash
./05_runtime/environments/edge-assistant/bin/python scripts/build_vision_b1.py \
  --role detection_segmentation --output 05_runtime/validation/b1-segmentation-new
./05_runtime/environments/edge-assistant/bin/python scripts/probe_segmentation_b1.py \
  --build-root 05_runtime/validation/b1-segmentation-new \
  --output 05_runtime/validation/b1-segmentation-new/parity
```

CPU 导出／参考环境为项目内 `05_runtime/builds/vision-export`，完整版本在 [requirements-export.lock](../01_src/deploy/vision/requirements-export.lock)。首次恢复可用 Python 3.12 创建该 venv，然后执行 `./05_runtime/builds/vision-export/bin/python -m pip install --no-deps -r 01_src/deploy/vision/requirements-export.lock`；`--no-deps` 用于保留完整锁文件指定的 headless OpenCV。应用环境不安装 Torch。需要独立复核 ONNX 导出时，将锁定的 `yolov8n-seg.pt` 复制到 `05_runtime/` 下的新目录，再用应用 Python 运行 `scripts/export_segmentation_b1.py --weights 新目录/yolov8n-seg.pt --output 新验证目录`。已有 ONNX 不会被覆盖；导出脚本只移除日期元数据并排序描述字段，检查推理图字节不变。新的项目内环境已独立重复导出，部署 ONNX 的 SHA-256 与原记录完全相同。

本次定位到一个参考环境问题：原工具环境使用 OpenCV 5.0、NumPy 2.5，而应用使用 OpenCV 4.11、NumPy 1.26。对同一输入张量，ONNX/TensorRT 已一致，但缩放图像存在一个灰度级的差异，影响了端到端框和 mask。固定参考环境后，原先的验收阈值全部通过；失败数据和修正记录均保留在可部署性报告中。后续优化比较必须同时固定模型、输入和预处理库版本。

## 头姿、视线与外观特征

新增部署候选及边界：

| 能力 | 已核验模型 | 接口与边界 |
|---|---|---|
| 头姿 | OMZ head-pose-estimation-adas-0001 | BGR 60×60；yaw/pitch/roll 度；采用模型公开的角度范围 |
| 视线辅助关键点 | OMZ facial-landmarks-35-adas-0002 | 同一人脸 ROI 的 35 点，用前四点构造左右眼区域；不对应私有 70／52 点定义 |
| 视线 | OMZ gaze-estimation-adas-0002 | 左眼、右眼及同帧头姿三路输入；包含 roll 对齐和补偿；没有估计闭眼状态，也没有标定到世界坐标 |
| 行人 ReID 特征 | OSNet x0.25 MSMT17 | RGB 256×128，512 维归一化特征；目标关联仍需接入 |
| 物体特征 | OpenAI CLIP ViT-B/32 图像编码器 | RGB 224×224，512 维特征；不把语义相似度当作通用物体身份认证 |

视线默认选用 OMZ 配套级联。L2CS 的 Gaze360 权重来源当前失效，且 Gaze360 的许可明确限制训练模型的商业应用；选择依据与来源保存在候选清单。6DRepNet 保留为未启用的替代候选。OpenVINO 只存在于隔离的转换／参考环境，正式应用使用 TensorRT。

当前所选模型的 ONNX 已准备。重建引擎和复核固定样本时，使用新的输出目录：

```bash
./05_runtime/environments/edge-assistant/bin/python scripts/build_vision_b1.py \
  --role head_pose gaze gaze_landmarks person_reid object_embedding \
  --output 05_runtime/validation/b1-attributes-new
./05_runtime/environments/edge-assistant/bin/python scripts/probe_remaining_vision_b1.py \
  --build-root 05_runtime/validation/b1-attributes-new \
  --attribute-root 05_runtime/validation/b1-attributes-new \
  --output 05_runtime/validation/b1-attributes-new/parity
```

该检查使用已准备的人脸检测和检测分割固定样本。它覆盖原始 PyTorch 外观特征、原始 OpenVINO FP32 头姿／关键点／视线，以及 TensorRT 完整级联；这是参考一致性，不能当作真实人脸身份错误率、视线真值精度或目标重识别准确率。

源模型与导出配方在候选清单的 `export_sources` 中。首次导出入口是 `scripts/export_remaining_vision_b1.py`，仅在目标 ONNX 不存在时运行。PyTorch 使用 `05_runtime/builds/vision-export`；OpenVINO 使用独立的 `05_runtime/builds/openvino-tools`，版本在 [requirements-openvino.lock](../01_src/deploy/vision/requirements-openvino.lock)。所有现有源模型和已验证图均保留，不覆盖旧的失败记录。

OMZ 转换中实际遇到两类兼容问题：TensorRT 不接收未使用的 MaxPool 索引输出，以及转换图未显式填写部分 Slice 的轴。`deploy/vision/onnx_compat.py` 使用 NVIDIA ONNX GraphSurgeon，只删除确认没有消费者的可选输出，并核对原始 StridedSlice 掩码后补齐轴；原始 IR 与修正 ONNX 的数值检查通过后才构建引擎。未改变精度或模型权重。

多模型运行器统一持有进程级 TensorRT logger，遵守其生命周期要求；已验证其余模型驻留时关闭、回收和重新加载头姿模型。整机并发、长期恢复与资源基线仍需要在 B1 统一管线完成后测量。

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
