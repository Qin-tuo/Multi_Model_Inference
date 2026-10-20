# B2：三个标准视觉模型替换

本次只替换人体／手部姿态、人脸特征、行人 ReID。沿用现有进程内 TensorRT 调度和四服务结构，另外七个视觉模型、4B、ASR、TTS、相机深度及手势规则保持原方案。

## 模型与接口

| 模块 | B1 | B2 | 接口变化 |
|---|---|---|---|
| 人体／手部姿态 | RTMW3D-X | 官方 RTMW-L 2D，Cocktail14，384×288 | 133 个 `(x,y)`，每手 21 点；删除模型预测的相对 z |
| 人脸特征 | SFace | ArcFace-MobileNetV1，官方 CASIA-WebFace v1.1 | 128 维改为 512 维；新旧特征库空间隔离 |
| 行人 ReID | OSNet x0.25 | 官方 OSNet x1.0，MSMT17 combineall | 仍为 512 维；特征空间随权重版本变化 |

D435i 继续提供对齐深度和目标距离。当前没有把每个关节点自动转换为米制三维坐标；二维关键点、相对 z、米制深度不能混用。

保留的七个模型是 YOLOv8n-seg、YuNet、OMZ 头姿／35 点／视线三个模型、ProgressiveTeacher 表情和 CLIP 物体特征。模型总数仍为 10。

源码、官方权重来源、完整哈希和预后处理契约见 [B2 来源清单](../03_models/candidates/vision-b2.json)；板端引擎见 [B2 运行清单](../03_models/vision-b2-runtime.json)。所有视觉网络使用 TensorRT FP32、关闭 TF32，本次不开展量化、频率或并发优化。

## 验证状态

三个模型均完成原生参考→ONNX→TensorRT 的有限固定输入核验，使用官方网络及预训练权重，不改学习层或重新训练。姿态预处理、坐标还原、人脸对齐和特征归一化也纳入核验。数值一致性不等于真人识别精度。

十模型完整回放、公开样本人脸登记／重开，以及完整张掌片段的停止事件已通过。原生 RGBD、文字／WAV→4B→TTS、整轮取消和服务清理的共存测试通过。十模块及七组整链路基线已测量，完成状态以 [替换进度记录](../benchmarks/results/b2-replacement-progress-20261020.json) 为准。

RTMPose-M 在已保存的真人张掌序列中未触发停止；RTMPose-L 虽能触发，时机和挥手事件数与旧方案不同。最终 RTMW-L 2D 在该张掌序列与主要挥手片段中复现了 B1 的对应事件帧。它们是局部回归证据，不能换算成召回率或准确率。

旧公开图片存在手指弯曲及手部裁切，不能作为严格张掌的无歧义正样本。旧失败记录全部保留。新 [张掌验收片段](../benchmarks/manifests/b2-held-palm-v1.json) 使用用户确认完成的实拍动作，独立检查首／中／尾帧，要求触发停止且不能触发挥手。全管线测试重新运行检测和姿态；其深度为明确标记的模拟值，回放时钟也不同于原始录像。没有修改手势门限来获得通过。

人脸匹配属于本地实验性标签功能，不参与权限验证。ArcFace 当前匹配阈值是开发策略，尚未进行现场 FAR／FRR 标定。旧 SFace 登记保留，但不能拿旧向量匹配新模型；B2 需要用户主动重新登记。

## B2 实测基线（2026-10-20）

源码固定在 `45af3ed`，复用 B1 R2 的五张固定输入、触发预算和七种负载。单模块每个输入预热 5 次，测量 20 组普通／带观察器配对调用；十模块共 440 组，输出一致性均通过。每个模块在独立进程中执行，四个服务停止。下表包含预处理、传输、引擎和后处理，单位毫秒。

| 模块 | P50 | P95 |
|---|---:|---:|
| 检测／分割（YOLOv8n-seg） | 27.93 | 34.18 |
| 人脸检测（YuNet） | 8.07 | 8.66 |
| 人体／手部（RTMW-L 2D） | 58.81 | 59.30 |
| 行人特征（OSNet x1.0） | 7.80 | 8.18 |
| 头姿 | 1.03 | 1.28 |
| 视线辅助关键点 | 2.63 | 2.68 |
| 视线 | 1.60 | 1.92 |
| 人脸特征（ArcFace-MobileNetV1） | 5.39 | 5.85 |
| 表情 | 3.76 | 3.81 |
| 物体特征（CLIP） | 20.56 | 20.90 |

整链路保持视觉处理上限 5 FPS。每组完成预热后记录 60 秒，每 100 ms 观察一次最新结果；下表延迟为整帧处理耗时，帧龄从主机接收时刻计算。可用内存为 1 秒采样中的板载最小值，CPU/GPU 共用内存。

| 条件 | 处理 FPS | P50 ms | P95 ms | P95 帧龄 ms | 最小可用 GiB |
|---|---:|---:|---:|---:|---:|
| 固定回放单跑 | 5.00 | 121.26 | 169.16 | 399.98 | 9.115 |
| 固定回放＋后端常驻 | 4.98 | 120.62 | 177.95 | 452.66 | 2.809 |
| 固定回放＋4B 推理 | 4.58 | 185.42 | 254.25 | 514.37 | 2.550 |
| 固定回放＋文件语音对话 | 4.37 | 194.96 | 299.60 | 592.32 | 2.425 |
| 固定回放＋带图问答 | 4.42 | 199.61 | 283.91 | 571.43 | 2.455 |
| D435i 实拍＋后端常驻 | 4.98 | 92.36 | 128.48 | 380.26 | 2.579 |
| D435i 实拍＋文件语音对话 | 4.25 | 223.26 | 292.72 | 573.67 | 2.387 |

七个计时窗口共完成 1,955 帧，整帧失败及观察器错过采样截止均为 0。视线适配器仍有拒绝／失败计数（含预热），见原始汇总；不能把零整帧失败表述为所有模型零拒绝。

两组实拍工作量不同：常驻组只有 14/299 帧执行姿态、2/299 帧执行面部属性；语音对话组分别为 255/255 和 244/255 帧。它们记录各自当时的实际距离门控和场景，不适合直接估计等负载并发开销。固定回放使用模拟对齐深度，实拍使用 D435i 原生深度；计时不含浏览器和 WebSocket 传输。

[单模块汇总](../benchmarks/results/b2-vision-modules-v1.json) · [整链路汇总](../benchmarks/results/b2-vision-stream-v1.json) · [冻结协议](../benchmarks/manifests/b2-vision-measurements-v1.json) · [负载定义](../benchmarks/manifests/b2-vision-stream-v1.json)。旧原始测量目录已于 2026-10-20 按用户要求删除，版本化汇总与冻结输入保留；仍需使用的参考输入迁入 `05_runtime/datasets/validation-inputs/`。原计时使用独立的已提交源码副本，原项目负责服务归属；复制回主仓后冻结校验再次通过。

复现时结束当前交互，并使用新输出目录：

```bash
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_modules.py \
  --protocol benchmarks/manifests/b2-vision-measurements-v1.json \
  --output 05_runtime/validation/my-b2-modules-01
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_stream.py \
  --workload benchmarks/manifests/b2-vision-stream-v1.json \
  --conditions replay-only --output 05_runtime/validation/my-b2-stream-01
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_stream_summary.py \
  --output 05_runtime/validation/my-b2-stream-01
```

## 尚未关闭的退出问题

正式计时前的一次停止操作中，`app` 和 `speech` 超过 systemd 退出时限，状态为 `timeout`；进程组随后为空，服务恢复 `READY`。后续完整测量的停止操作均正常，但根因尚未确认，编号 `B2-EXIT-01`。保留原退出时限和失败记录，后续单独定位；本轮不宣称长期稳定性验收完成。证据见 [进度记录中的 open_issues](../benchmarks/results/b2-replacement-progress-20261020.json)。

## 运行与回退

`run.sh` 默认选择 B2，已有环境变量和显式配置仍优先。切换时先结束当前轮次：

```bash
./run.sh services down
./run.sh services up --config 02_config/settings.orin-nx-b2.yaml
./run.sh services status
```

后端实现标识仍为 `tensorrt_b1`，B2 复用同一调度实现；实际模型由 `perception.model_manifest` 选择，状态中的 `model_set` 为 `B2-standard-2d`。

回到 B1：

```bash
./run.sh services down
./run.sh services up --config 02_config/settings.orin-nx-b1.yaml
```

历史 B1 冻结基线需要对应的旧源码；仅切换配置不等于恢复旧代码。替换前恢复点为 `before-upstream-standard-20261020`。权重、引擎和原始录像留在本机 `05_runtime/`，普通 Git 不包含它们；已有 B1 R2 运行包不包含新增 B2 模型。

## 阅读入口

1. `03_models/candidates/vision-b2.json`：网络来源、输入输出、预后处理和特征空间。
2. `app/vision_models.py`：根据清单选择适配器并校验引擎契约。
3. `app/wholebody_pose2d.py`、`app/face_tasks.py`、`app/appearance.py`：推理输入、输出及坐标／向量语义。
4. `app/identity_gallery.py`、`app/visual_tracks.py`：隔离不同模型的特征空间。
5. `scripts/export_standard_vision.py`、`scripts/build_standard_vision.py`：原生模型导出、固定输入核验和独占构建。

`app/` 相对 `01_src/`。新基线应与 B1 分开保存；跨模型的速度差异不算同一模型的优化收益。后续性能优化仍由用户在指导下实施。
