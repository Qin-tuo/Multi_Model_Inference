# 视觉基准与复现

本轮保持模型和参数不变，已完成整合前后同输入对照。以下先列当前基线，再保留原 B2 数字及复现方法。

## 2026-10-21 整合后的测量

代码提交 `714290e`，运行源码按 `71842c6` 冻结；模型、输入、数值参数和测试负载与原 B2 对齐。七组各 60 秒，共 1,997 帧，整帧失败与采样截止时间遗漏均为 0。这里是重构后的基线，未进行量化或调度优化。

| 模型角色 | 原 B2 P50 / ms | 当前 P50 / ms | 当前 P95 / ms |
|---|---:|---:|---:|
| `detection_segmentation` | 27.926 | 27.879 | 34.081 |
| `face` | 8.068 | 8.087 | 8.890 |
| `body_hand_pose` | 58.812 | 58.728 | 59.174 |
| `person_reid` | 7.795 | 7.640 | 8.269 |
| `head_pose` | 1.027 | 1.513 | 1.736 |
| `gaze_landmarks` | 2.634 | 2.628 | 2.654 |
| `gaze` | 1.599 | 1.607 | 1.708 |
| `face_identity` | 5.391 | 5.222 | 5.631 |
| `expression` | 3.762 | 3.766 | 3.908 |
| `object_embedding` | 20.563 | 20.604 | 20.858 |

| 条件 | 实际完成 FPS | 处理 P50 / P95，ms | 最新结果帧龄 P95，ms | 最低可用 RAM，GiB |
|---|---:|---:|---:|---:|
| `replay-only` | 4.98 | 121.40 / 174.99 | 442.68 | 9.864 |
| `replay-resident` | 4.98 | 123.54 / 167.69 | 410.42 | 3.317 |
| `replay-4b` | 4.57 | 185.73 / 265.52 | 532.60 | 3.229 |
| `replay-conversation` | 4.38 | 199.22 / 296.27 | 577.40 | 3.092 |
| `replay-vision-chat` | 4.43 | 199.94 / 284.38 | 573.22 | 3.138 |
| `live-resident` | 4.98 | 87.24 / 103.07 | 374.61 | 3.488 |
| `live-conversation` | 4.95 | 118.10 / 146.45 | 396.46 | 3.097 |

两项差异做了补充调查，初测数据仍保留：头姿初测约 1.51 ms，旧／新／新／旧复测分别约 1.51／1.02／1.01／1.03 ms，偏慢情形也出现在旧源码；仅视觉回放的帧龄 P95 原记录为 399.98 ms，当前初测 442.68 ms，补充旧／新复测为 441.22／443.76 ms。未复现这两项明显的版本特有退化，但尚未证明波动的具体硬件或调度原因。

固定回放的 FPS 变化在约 ±0.4% 内；初测 `replay-4b` 处理 P95 增加约 4.4%，如实保留。一次 60 秒窗口没有统计置信区间，不能宣称性能严格等价。此次两个 live 窗口的人体姿态、人脸属性调用为零，不能与旧有人场景计算加速比，也不代表十模型全负载性能。

[当前单模型结果](../benchmarks/results/vision-modules.json) · [当前整链路结果](../benchmarks/results/vision-stream.json) · [逐项对照与差异调查](../benchmarks/results/consolidation-comparison.json)


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

默认生成 `benchmarks/manifests/vision-measurements.json` 与 `model-measurements.json`，输入目录为 `05_runtime/baselines/current-vision/`。整链路负载在 `benchmarks/manifests/vision-stream.json` 固定条件、时长、输入和背景请求。当前冻结与测量已经完成；重新实验时使用新的输出目录，不覆盖已有协议或结果。

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

当前界面基准数据文件为 `benchmarks/results/vision-modules.json`，其 `runtime_source_commit` 和时间随结果显示。该文件已更新为本轮实测结果，整合前 B2 的版本化文件保持原字节。退出超时 `B2-EXIT-01`、未取得的真人/实体声学与长稳证据单独保留，不用新的耗时数字替代功能验收。
