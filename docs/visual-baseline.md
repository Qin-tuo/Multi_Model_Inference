# 视觉基准与复现

当前模型和参数沿用整合前 B2。本轮完成代码整理后，必须重新冻结当前源码并用相同输入、模型和测量规则对照；在新结果完成前，下列数字只代表整合前的已测 B2，不能宣称为整合后的新性能。

## 已保留的整合前 B2

原始源码为 `45af3ed`，单模型使用固定 640×480 图片及各自的固定 ROI；预热后普通调用与插桩调用交替配对。各阶段分位数不能直接相加。固定回放使用明确标记的模拟 1.5 米深度，真实 D435i 窗口单独报告。

| 模型角色 | 普通适配器调用 P50 / ms | P95 / ms |
|---|---:|---:|
| `detection_segmentation` | 27.926 | 34.178 |
| `face` | 8.068 | 8.662 |
| `body_hand_pose` | 58.812 | 59.296 |
| `person_reid` | 7.795 | 8.176 |
| `head_pose` | 1.027 | 1.279 |
| `gaze_landmarks` | 2.634 | 2.681 |
| `gaze` | 1.599 | 1.915 |
| `face_identity` | 5.391 | 5.853 |
| `expression` | 3.762 | 3.813 |
| `object_embedding` | 20.563 | 20.896 |

| 条件 | 实际处理 FPS | 整帧处理 P50 / P95，ms | 最新结果帧龄 P95，ms |
|---|---:|---:|---:|
| `replay-only` | 5.00 | 121.26 / 169.16 | 399.98 |
| `replay-resident` | 4.98 | 120.62 / 177.95 | 452.66 |
| `replay-4b` | 4.58 | 185.42 / 254.25 | 514.37 |
| `replay-conversation` | 4.37 | 194.96 / 299.60 | 592.32 |
| `replay-vision-chat` | 4.42 | 199.61 / 283.91 | 571.43 |
| `live-resident` | 4.98 | 92.36 / 128.48 | 380.26 |
| `live-conversation` | 4.25 | 223.26 / 292.72 | 573.67 |

[完整单模块结果](../benchmarks/results/b2-vision-modules-v1.json) · [完整链路结果](../benchmarks/results/b2-vision-stream-v1.json) · [原协议](../benchmarks/manifests/b2-vision-measurements-v1.json) · [原负载定义](../benchmarks/manifests/b2-vision-stream-v1.json)

此前清理过的 B2 原始测量已从冻结副本恢复到 `benchmarks/raw/b2-before-consolidation/`，此目录不进入普通 Git；相同原始文件已逐一校验复制到 `05_runtime/baselines/b2-original-measurements/`，供独立运行包携带。固定输入在 `05_runtime/baselines/b2-vision-v1/`。必要模型参考样本在 `05_runtime/datasets/validation-inputs/`；标签、像素和旧协议不因改目录而重写。

两组 live 的人体/人脸触发次数不同，不能用它们直接计算同负载并发开销。整帧失败为零也不表示所有属性有效；视线输入质量拒绝与 TensorRT 故障分开解释。`capture_fps`、处理完成计数 FPS、延迟和帧龄是不同指标，配置的 5 FPS 上限不是实测 FPS。

## 冻结当前源码与输入

以下命令使用项目 Python 环境。先提交运行源码、配置和模型清单，再冻结；已存在的协议和输出目录不会被覆盖。

```bash
source ./05_runtime/environments/edge-assistant/bin/activate
python scripts/benchmark.py freeze --inputs-from benchmarks/manifests/b2-vision-measurements-v1.json
python scripts/benchmark.py freeze-models
```

默认生成 `benchmarks/manifests/vision-measurements.json` 与 `model-measurements.json`，输入目录为 `05_runtime/baselines/current-vision/`。整链路负载在 `benchmarks/manifests/vision-stream.json` 固定条件、时长、输入和背景请求。当前整合的冻结/测量完成状态以执行清单为准。

## 单模型、整链路与只读观察

```bash
python scripts/benchmark.py vision --output 05_runtime/validation/vision-modules-new
python scripts/benchmark.py stream --output 05_runtime/validation/vision-stream-new
python scripts/benchmark.py stream-summary --output 05_runtime/validation/vision-stream-new
python scripts/benchmark.py live --seconds 120 --condition resident-idle --note "固定可见场景、对话空闲" --output 05_runtime/validation/live-observation-new
```

前两个测量会协调本项目服务并保留前后快照。单模型逐个隔离引擎；整链路使用最新帧管线和固定的驻留/4B/语音/带图压力条件。记录预热、失败、取消、未完成请求、任务触发、主机/进程资源及板端采样，不筛掉不利样本。

`live` 只读取已有服务，没有生成测试负载。它按帧号去重并核对完成计数，漏采、重启和 HTTP 异常均标出；只能代表被观察的窗口。调试面板同样不替代正式基准。

## 定位与历史复现

```bash
python scripts/benchmark.py vision --role detection_segmentation --nsys --output 05_runtime/validation/vision-profile-new
```

带 Nsight/NVTX 的运行是诊断，不能作为普通延迟基准。查看同一输入的 CPU 预处理、传输、CUDA 执行、后处理和等待，再设计单因素实验。性能优化和任务质量评测由用户主导，具体步骤见 [人体检测实操](body-detection-optimization.md)。

旧协议要求对应的历史源码；当前协调器支持 `vision --source-root <匹配的冻结源码目录> --protocol <旧协议>`，并从原项目管理服务。不要修改旧协议的哈希来绕过校验。历史源码及报告的恢复点为 `before-mainline-consolidation-20261020` 和 `before-historical-file-retirement-20261021`。

当前界面基准数据文件为 `benchmarks/results/vision-modules.json`，其 `runtime_source_commit` 和时间随结果显示。完成新测量后更新该文件，整合前 B2 的版本化文件仍保持原字节。退出超时 `B2-EXIT-01`、未取得的真人/实体声学与长稳证据单独保留，不用新的耗时数字替代功能验收。
