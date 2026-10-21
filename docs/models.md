# 当前模型与接口边界

当前十个视觉网络均由应用内的 TensorRT 后端驱动，使用已锁定的 FP32 引擎并关闭 TF32。D435i 原生深度由相机与 SDK 提供，不计为神经网络模型。本轮只整理代码与工件，不更换权重、精度、预后处理或调度参数。

| 模块角色 | 当前公开模型 | 对外能力 |
|---|---|---|
| `detection_segmentation` | YOLOv8n-seg COCO | 人体/物体检测与实例分割 |
| `face` | YuNet 2023mar | 人脸框与五点 |
| `body_hand_pose` | RTMW-l Cocktail14 COCO-WholeBody 2D 133, 384x288 | 133 个二维全身关键点，含双手 |
| `person_reid` | OSNet x1.0 MSMT17 combineall official | 512 维行人外观特征 |
| `face_identity` | ArcFace MobileNetV1 official CASIA-WebFace v1.1 | 512 维人脸特征与可选匹配 |
| `expression` | Progressive Teacher MobileFaceNet 2022july | 公开模型的表情类别与分值 |
| `head_pose` | head-pose-estimation-adas-0001 | 头部姿态角 |
| `gaze_landmarks` | facial-landmarks-35-adas-0002 | 视线网络需要的面部关键点 |
| `gaze` | gaze-estimation-adas-0002 | 视线向量/角度 |
| `object_embedding` | OpenAI CLIP ViT-B/32 image encoder | 物体图像特征 |

来源、固定提交、权重哈希、模型卡、许可证和预后处理契约见 [vision-models.json](../03_models/vision-models.json)；实际引擎哈希和输入形状见 [vision-runtime.json](../03_models/vision-runtime.json)。共有十个网络，因为头姿、视线辅助关键点和视线估计是三个独立的级联模型。

人体/手部使用 RTMW-L 二维姿态。它不输出原定制 70 点模型的相对 z；D435i 距离也不冒充模型预测的 z。手势由人体模型中的手部关键点和时序规则得到，当前定义为挥手问候、静止张掌取消整轮。原环境自定义的完整十类手势、70/52 点协议和 `exaggeration` 分值没有被声称为等价复现。

人脸特征使用 ArcFace-MobileNetV1，行人特征使用 OSNet x1.0。特征空间绑定模型哈希，旧图库与新特征不混用。物体特征表达外观相似性，不能据此保证任意物体的永久身份识别。

语言模型为 Qwen3.5-4B Q4_K_M 与 F16 视觉投影，由 llama.cpp 提供 OpenAI 兼容接口；ASR 为 Qwen3-ASR-0.6B，TTS 为 OpenVoiceStream 的 Matcha/Vocos。语音与语言模型的来源及文件摘要在 [manifest.json](../03_models/manifest.json)，VAD 和语音接口在 [speech.json](../03_models/speech.json)。本轮仍保留这些后端。

## 验证与后续替换

`python scripts/build.py artifacts --check-only` 校验当前视觉文件。原生参考入口为 `scripts/validate.py` 的 `face`、`expression`、`segmentation`、`attributes`、`standard-models` 任务；相关导出、原生框架及参考样本属于开发依赖，需要相应的隔离环境。它们不是运行时推理回退。

`python scripts/build.py engines` 构建候选引擎并在输出目录写入候选清单；导出任务也只写候选记录。后续模型替换需重新检查形状、坐标、预后处理、任务质量、资源占用与图库特征空间，再完成管线和取消回归。接口一致不等于模型能力已等价。

整合前的三模型替换记录保存在 [B2 证据](../benchmarks/evidence/b2-replacement-progress-20261020.json)。当前阶段完成度以 [执行清单](../benchmarks/manifests/consolidation-progress.json) 为准。
