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

十模型完整回放、公开样本人脸登记／重开，以及完整张掌片段的停止事件已通过。原生 RGBD、文字／WAV→4B→TTS、整轮取消和服务清理的共存测试通过。新性能基线待测量，完成状态以 [替换进度记录](../benchmarks/results/b2-replacement-progress-20261020.json) 为准。

RTMPose-M 在已保存的真人张掌序列中未触发停止；RTMPose-L 虽能触发，时机和挥手事件数与旧方案不同。最终 RTMW-L 2D 在该张掌序列与主要挥手片段中复现了 B1 的对应事件帧。它们是局部回归证据，不能换算成召回率或准确率。

旧公开图片存在手指弯曲及手部裁切，不能作为严格张掌的无歧义正样本。旧失败记录全部保留。新 [张掌验收片段](../benchmarks/manifests/b2-held-palm-v1.json) 使用用户确认完成的实拍动作，独立检查首／中／尾帧，要求触发停止且不能触发挥手。全管线测试重新运行检测和姿态；其深度为明确标记的模拟值，回放时钟也不同于原始录像。没有修改手势门限来获得通过。

人脸匹配属于本地实验性标签功能，不参与权限验证。ArcFace 当前匹配阈值是开发策略，尚未进行现场 FAR／FRR 标定。旧 SFace 登记保留，但不能拿旧向量匹配新模型；B2 需要用户主动重新登记。

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
