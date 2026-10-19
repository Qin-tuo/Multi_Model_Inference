# 视觉基线复现入口

## B1 模型接入验证

2026-10-19：B1 尚未冻结，默认服务仍使用 B0 配置。按用户最新决定，本阶段深度直接采用 D435i 原生深度和 RealSense SDK 对齐，不再要求神经网络深度推理。9 类所选视觉神经模型及 1 个视线辅助模型已完成有限固定输入的 TensorRT 核验，统一管线也已执行全部 10 个模型。模型证据见 [可部署性记录](../benchmarks/results/b1-feasibility-20261019.json)，集成证据见 [集成记录](../benchmarks/results/b1-integration-20261019.json)。RTMW3D 原始权重与预后处理核验现已通过；真实场景覆盖、长期恢复、B1 性能和部署交接仍未完成。

### 集成结构与当前资源边界

独立配置是 [settings.orin-nx-b1.yaml](../02_config/settings.orin-nx-b1.yaml)，启动仍走现有四个 systemd 服务。视觉只有一个最新帧工作线程；每个模型持有一个执行上下文和固定形状缓冲区，按顺序执行，未引入额外推理服务。

| 文件 | 职责 |
|---|---|
| `01_src/app/perception_b1.py` | 同一组 RGBD 的检测、距离门控、人物/物体/人脸关联、按目标调用模型与结果发布 |
| `01_src/app/vision_models.py` | 加载 10 个锁定引擎、核验版本与模型语义、统计调用、统一释放 |
| `01_src/app/wholebody_pose.py` | 复用固定 MMPose SimCC 解码，输出 133 点与左右手 21 点；相对 z 不当作米制距离 |
| `01_src/app/visual_tracks.py` | Norfair 运动/外观关联，限定目标数、有效期和源标识；不自行实现跟踪算法 |
| `01_src/app/identity_gallery.py` | SQLAlchemy 本地人脸标签库，显式登记、未知/歧义结果和模型空间隔离 |
| `03_models/vision-b1-runtime.json` | 当前板端引擎、源图与语义契约校验值；仍标为集成配置而非已验收基线 |

检测与分割每帧最多 10 个目标；当前配置处理最近的 1 个有效距离内人物、1 个物体特征目标及最多 2 张人脸属性。距离门限为 0.35–3 m，使用同帧对齐原生深度的目标 mask/ROI 中位数；无效深度不触发后续任务。实例 mask 供本帧测距使用，不长期缓存；网页当前展示框、距离、目标编号、手部关键点和人脸属性。显示未触发与模型异常的含义需要分别理解。

Norfair 目标最多 20 个，按真实接收时间在 2 秒后失效，重连更换源标识；预测位置不作为新检测发布。网页可选择当前观察目标，目标丢失后不会自动替换为另一个人/物。外观特征只对选定预算内目标计算，不能据此宣称多人交叉或长时遮挡重识别已通过。人脸登记最多 32 个标签，仅保存特征和模型空间，不保存人像，也不控制对话授权。

关键点输入已接回现有手势规则。SimCC 峰值允许大于 1，不能按概率裁剪；同一手的连续证据同时绑定人物编号。公开静止张掌图片连续回放已触发 stop，软件规则覆盖换人、低置信度和断帧，但不替代真实挥手、遮挡和多手补测。

运行依赖是在根目录 `requirements.lock` 上增加 [vision/requirements-runtime.lock](../01_src/deploy/vision/requirements-runtime.lock)；恢复时使用两个锁文件共同约束，避免升级已有 NumPy/OpenCV。Norfair 2.3.0 使用 BSD-3-Clause，FilterPy 使用 MIT；生产环境不加载 Torch/OpenVINO 参考模型。

固定公开图像回放使用真实 TensorRT 模型和明确标注的均匀 1.5 m 模拟深度，仅验证管线、数据归属、登记持久化和规则触发。完整服务探针使用真实 D435i，并执行文字→4B→TTS、生成的 WAV→VAD/ASR→4B→TTS 和取消操作。当前镜头无人物，因此后者仅覆盖场景检测/物体特征与对话共存。所有命令自动恢复之前的服务配置，输出目录必须是新目录：

```bash
./05_runtime/environments/edge-assistant/bin/python scripts/probe_b1_pipeline.py \
  --output 05_runtime/validation/b1-pipeline-new
./05_runtime/environments/edge-assistant/bin/python scripts/probe_b1_services.py \
  --output 05_runtime/validation/b1-services-new
```

完整服务探针可加 `--webdriver http://127.0.0.1:4445`，连接已运行的本地 Firefox WebDriver，检查相机页和 `/debug`。浏览器运行在板上会额外占用资源，这些功能测试读数不是 B1 性能结论。正式基线将固定输入、预热、负载和采样条件；现阶段不开展量化/并发调优。

### 恢复与测量流程

`scripts/probe_b1_recovery.py --output 新目录` 先用公开人物图片和显式模拟深度验证距离门控，再执行三轮真实 D435i SDK 关闭/重开、单次推理异常注入、恢复和全部模型释放。过近、过远、无效或缺失深度时，只运行基础检测；过期/旧源帧在推理前被拒绝，推理途中失效的帧不能发布，人脸登记和目标选择也拒绝过期观察。这不等于物理 USB 拔插或长期稳定性验收。

模型重新加载会重置耗时样本，跟踪重置会清空帧顺序和目标状态。重复加载被拒绝，保留现有资源归属。释放过程中即使某个模型报错，仍尝试关闭其他资源；对于底层辅助代码会吞掉的 CUDA 释放错误，适配器检查缓冲区/流的真实状态，未确认释放就保留归属并禁止后续推理，允许再次关闭。

正式单模块测量入口为 `01_src/benchmarks/vision_b1_modules.py`。准备程序 `scripts/freeze_b1_inputs.py` 要求代码已提交，先把公开图片固定为 640×480、保存模型产生的 ROI 参数，并在真实 TensorRT 上验证普通调用与计时调用输出一致，再写入不可覆盖的输入/运行代码协议。每个模块在独立进程中加载一个引擎，按固定输入预热 5 次，普通/计时调用交替执行 20 对；保存每次原始耗时、预测样本、资源数据和执行源码。

计时只在测量进程中安装，继续调用项目锁定的 NVIDIA 推理辅助代码。主机预处理、同步原生调用和后处理属于连续墙钟区间；H2D、引擎执行、D2H 由同一 CUDA 流上的事件计时，包含可能的主机提交间隙，不等于单个 kernel 的执行时间，也不能与主机区间相加。配对数据用于展示观测开销，完整管线与 4B/语音共存数据另测，B0 原始记录保持不变。

### B1 单模块初始数据

2026-10-19 已完成 10 个模块、22 组固定输入的测量，共 440 对普通/计时调用，全部输出一致；资源采样无错误。测量为 Orin NX 16GB、MAXN 模式、FP32/关闭 TF32，每个进程只加载一个视觉引擎，4B/语音服务停止。下表是每模块不同固定输入合并后的普通调用耗时，包含该适配器的预处理和后处理；检测/人脸包含黑色负样本，各输入分别统计的数据保留在 [完整汇总](../benchmarks/results/b1-vision-modules-summary.json)。

| 模块 | P50 / ms | P95 / ms |
|---|---:|---:|
| 检测与实例分割 | 28.55 | 39.35 |
| 人脸检测 | 8.02 | 8.77 |
| 人体与手部关键点 | 88.53 | 89.06 |
| 行人特征 | 4.05 | 4.66 |
| 头部姿态 | 1.02 | 1.25 |
| 视线辅助关键点 | 2.64 | 2.68 |
| 视线 | 1.59 | 1.65 |
| 人脸特征 | 5.74 | 6.39 |
| 表情 | 3.32 | 3.41 |
| 物体特征 | 20.54 | 20.81 |

这些数据不代表整机帧率，各行或不同阶段的分位数不能直接相加。输入与运行代码契约在 [冻结协议](../benchmarks/manifests/b1-vision-measurements-v1.json)，协议记录的是冻结时状态；当前进度以 B1 验收清单为准。实际测量脚本源码、每次原始耗时、CUDA 事件区间、预测样本、进程/系统资源和 tegrastats 都保存在 `05_runtime/validation/b1-modules-20261019/`。完整管线数据见下一节，持续稳定性及部署交接仍待完成。

```bash
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_modules.py \
  --output 05_runtime/validation/b1-modules-new
# 已有结果重新汇总，无推理和服务切换：
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_modules.py \
  --output 05_runtime/validation/b1-modules-20261019 --summarize
```

### B1 连续视觉与共存数据

已完成 [固定共存协议](../benchmarks/manifests/b1-vision-stream-v1.json) 的 7 组测量，每组预热后测 60 秒，共 4,200 次周期采样，均有有效结果，无整帧处理失败、过期丢弃或后端进程重启。采集目标 10 FPS，处理目标上限 5 FPS；最新帧策略主动跳过部分采集帧以避免积压，这与异常掉帧分开计数。全部原始结果见 [整链路汇总](../benchmarks/results/b1-vision-stream-summary.json)。

下表前五行使用相同的五份固定图片、模拟对齐深度和相同的回放时序。后两行为 D435i 实拍，当时没有人体/人脸进入画面，条件触发的任务较少，不能与前五行直接比较加速比。

| 条件 | 处理 FPS | 整帧处理 P50 / P95，ms | 最新结果帧龄 P95，ms | 系统最低可用内存，GiB |
|---|---:|---:|---:|---:|
| 回放，仅视觉 | 4.95 | 150.45 / 201.69 | 428.40 | 9.27 |
| 回放，4B/语音后端驻留 | 4.93 | 151.58 / 205.66 | 439.62 | 2.54 |
| 回放，4B 连续生成 | 3.95 | 238.73 / 315.62 | 622.90 | 2.46 |
| 回放，文件语音→ASR→4B→TTS | 3.98 | 226.42 / 335.95 | 649.62 | 2.34 |
| 回放，带图问答→TTS | 3.88 | 242.23 / 336.24 | 656.37 | 2.35 |
| 实拍，后端驻留 | 4.98 | 43.76 / 61.74 | 348.77 | 2.55 |
| 实拍，完整文件语音请求 | 4.98 | 63.01 / 79.47 | 365.73 | 2.33 |

帧龄来自接收单调时钟，不含传感器曝光到主机接收的时间；周期采样会观察到一帧在下次结果产生前逐渐变旧，因此它与单次处理耗时不同。原始数据同时给出接收→开始、JPEG 完成→开始、接收→完成和各阶段耗时。GUI/WebSocket 传输未计入本次性能窗口，相关功能已由独立服务/浏览器探针验证。

所有后端驻留条件都加载相同的客户端、VAD、动作库和 TurnController；请求负载使用隔离存储，逐轮清空会话，但保留服务器原有提示缓存策略。回放语音组有 20 个完整请求落在测量窗口内，请求 P50/P95 为 2.97/3.15 秒；实拍语音组为 25 个、2.35/2.41 秒。带图问答组只有 7 个完整请求，P50/P95 为 7.78/8.57 秒，样本数较少，不能据此作长期 SLA 承诺。窗口边缘及终止时取消的请求保留原始记录，不混入这些完整请求统计；各阶段、排队时间和缓存 token 数也已汇总。

整帧失败为零不代表每个属性均可用：回放中的部分视线适配器调用被拒绝，原始累计错误计数保留在 `model_adapter_counters_total_including_warmup`，需要与输入质量检查和模型执行故障区分。真人准确率、手势补测和长期稳定性依旧按原边界单列。板端 GPU 负载为 1 秒采样；当前 tegrastats 未提供 GPU MHz 字段，汇总保留缺失标记，不推算频率。

```bash
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_stream.py \
  --output 05_runtime/validation/b1-stream-new
# 仅重新读取原始数据，不启动模型：
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_stream_summary.py \
  --output 05_runtime/validation/b1-stream-20261019
```

### RTMW3D 原始权重与裁剪核验

原始权重采用 [MMPose 固定版本模型页](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/projects/rtmpose3d/README.md) 所列的 RTMW3D-X，完整 SHA-256 为 `b0a0eab7ca501e2c88028fe1cfd78b50c7130e220d39541da7f669e5de25e6ca`。权重地址、本地路径和大小记录在候选清单的 `body_hand_pose.source_checkpoint`。项目内已准备该文件与固定版本 MMPose 仓库。

参考进程直接加载上游 CSPNeXt、CSPNeXtPAFPN、RTMW3DHead，837 项状态严格加载，无缺失或多余键；原始预处理使用 GetBBoxCenterScale、TopdownAffine、PoseDataPreprocessor，解码使用 SimCC3DLabel。仅绕过会额外导入训练数据集的包聚合入口，不替换网络计算或编造缺失算子；所有实际导入的源文件校验值保存在 `native-provenance.json`。参考进程为 CPU PyTorch，应用仍使用 TensorRT。

四个公开人物/镜像/多人裁剪样本上，ONNX/TensorRT 相对原始权重的最大张量差为 `1.16e-5`，三轴取峰位置无差异；最终原图关键点最大数值误差约 `6.11e-5` 像素，相对 z 一致。这是参考实现一致性，不是人体关键点真实精度。

此次核验发现并修复了一个预处理问题：手写仿射矩阵与上游 float32 数组运算及 `get_warp_matrix` 的舍入不同，一个小数人物框有 129 个输入值变化，造成约 0.826 像素的关键点变化。正式适配器现复用上游纯 NumPy/OpenCV 几何函数，修复后四个输入张量逐值相同，验收阈值未放宽。失败原始记录和修复后的记录都保留，另有独立的合成渐变图张量哈希回归测试。

```bash
./05_runtime/environments/edge-assistant/bin/python scripts/probe_pose_source_b1.py \
  --output 05_runtime/validation/b1-pose-source-new
```

该探针临时停止并恢复原有项目服务，原始 CPU 参考和 ONNX 参考分进程执行，随后验证 TensorRT。参考环境锁文件增加了 MMEngine/MMCV-lite，并将 ContourPy、ml-dtypes 固定到支持 NumPy 1.26 的版本；仅供可视化/ONNX 类型支持的这些依赖不进入生产视觉环境。`pip check` 对参考环境仍会报告上游要求的 `opencv-python` 名称未安装，因为这里明确使用同版本 `opencv-python-headless`，不在一个环境中混装两个 OpenCV 发行包。

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
| 行人 ReID 特征 | OSNet x0.25 MSMT17 | RGB 256×128，512 维归一化特征；已接 Norfair 有界关联，真实遮挡/交叉仍需验证 |
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
