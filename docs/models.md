# 现用模型

模型来源和输入输出契约在 `03_models/vision-models.json`，引擎路径、SHA256、输入形状和精度在 `03_models/vision-runtime.json`。视觉引擎统一放在 `05_runtime/models/vision/`，均通过 TensorRT 推理。适配器位于 `01_src/app/vision/models/`，由 `manager.py` 统一加载；底层执行与工件校验位于 `01_src/app/inference/`。

运行清单 schema 2 还保存引擎实际的有序 I/O 名称、形状和类型，供原生图节点映射。模型适配器提供 `prepare` / `decode` 两个独立阶段；默认 `infer` 仍串联现有同步后端。图模式可注入不分配 GPU 资源的 `EngineDescription`，由 Holoscan 持有推理引擎。当前正式视觉管线尚未切换到新图，迁移状态见 `project-overview.md`。

| 角色 | 模型 | 主要输出 |
|---|---|---|
| 检测与分割 | YOLOv8n-seg | 类别、框、置信度和实例 mask |
| 人脸检测 | YuNet | 人脸框、置信度和 5 点关键点 |
| 人体/双手姿态 | RTMW-l，384×288 | COCO-WholeBody 133 个二维关键点，含双手 |
| 行人特征 | OSNet x1.0 | 512 维特征 |
| 头部姿态 | OMZ head-pose-estimation-adas-0001 | yaw / pitch / roll |
| 视线辅助点 | OMZ facial-landmarks-35-adas-0002 | 35 点面部关键点 |
| 视线 | OMZ gaze-estimation-adas-0002 | 视线向量及角度 |
| 人脸特征 | ArcFace MobileNet V1 | 512 维特征，用于可选图库匹配 |
| 表情 | ProgressiveTeacher MobileFaceNet | 7 类表情分值 |
| 物体特征 | CLIP ViT-B/32 | 512 维视觉特征 |

当前视觉引擎采用既定 FP32 设置。距离来自 D435i 原生深度；姿态模型只输出二维坐标，视线不是经过标定的世界坐标射线，表情分值不等同于真实情绪判定。

| 其他能力 | 当前后端与模型 |
|---|---|
| 图文/语言理解 | llama.cpp；Qwen3.5-4B Q4_K_M，F16 视觉投影 |
| 语音识别 | llama.cpp；Qwen3-ASR-0.6B Q8_0，BF16 音频投影 |
| 语音合成 | OpenVoiceStream / VoxEdge；Matcha BF16 TensorRT、Vocos FP16 TensorRT，保留其 ONNX 编码器与词典 |
| 语音活动检测 | Silero VAD v6，CPU ONNX Runtime |

语言、语音模型的文件摘要在 `03_models/manifest.json` 和 `03_models/speech.json`。语音框架保留实际服务代码与配置，使用 `01_src/deploy/speech/source-lock.json` 逐文件校验。所有运行文件都在本项目或主机系统库中。

当前目录不保留候选模型、未使用的原始 ONNX、导出 checkpoint、旧引擎、基准样本和模型备份。替换模型时必须同时满足适配器、输入输出契约及引擎清单；仅替换文件名不能完成兼容替换。
