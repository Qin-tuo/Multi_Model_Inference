# 人体检测完整优化实操

以已经完成引擎初测的 YOLOv8n-seg 为切入口，完整实践部署效率优化与模型压缩。每个候选都经过测量、定位、方案选择、实施、质量回归、板端性能对照和多模型共存验证，最终形成可以复现的优化案例。

用户亲自执行与解释实验，助手每轮提供一个明确问题及通常 1～3 条命令或一处改动。至少走通一项部署实现优化、一组 INT8 量化候选，以及一组结构化剪枝与微调候选的导出、部署和评测流程；具体方案是否保留，由实际效果与成本决定。

## 实验对象

本案例固定刚才测过的模型与原始记录。系统后续升级不会覆盖这个起点；若更换模型、输入尺寸或任务范围，另列方案与基线。

| 工件 | 项目位置 |
|---|---|
| 原始浮点权重 | `05_runtime/models/b1/detection_segmentation/yolov8n-seg.pt` |
| 部署 ONNX | `05_runtime/models/b1/detection_segmentation/yolov8n-seg-deploy.onnx` |
| 原始 FP32 引擎 | `05_runtime/models/b1/runtime/detection_segmentation.fp32.engine` |
| 来源与导出契约 | [vision-b1.json](../03_models/candidates/vision-b1.json) 中的 `detection_segmentation` |
| 引擎与版本锁 | [vision-b1-runtime.json](../03_models/vision-b1-runtime.json) |
| 预后处理入口 | [segmentation.py](../01_src/app/segmentation.py) |
| TensorRT 执行入口 | [tensorrt_runner.py](../01_src/app/tensorrt_runner.py) |

2026-10-20 已核对上述 `.pt`、部署 ONNX 和 FP32 引擎的哈希。原始权重可作为量化、剪枝和微调的起点，原始引擎继续保留作性能对照。

当前模型同时提供 80 类检测与实例 mask。人体类别是主学习指标；接回现有项目时，还要回归仍被使用的其他类别和 mask，因为它们参与测距与后续任务触发。学习阶段先保持任务与输出契约，纯人体检测或取消 mask 属于另一个方案。

## 设备分工

| 位置 | 主要工作 |
|---|---|
| 有 NVIDIA 独显的电脑或服务器 | 数据准备、训练框架质量评测、剪枝、微调；按所选工具进行量化校准或 QAT；导出 ONNX；查看性能报告 |
| Jetson | 按目标环境构建 TensorRT 引擎，核对转换后的效果，测耗时、内存、功耗、温度和多模型共存 |

面向 Jetson 的引擎在目标板上构建。训练机生成的 ONNX 等中间工件迁入板端后，核对形状、算子和实际计算精度。最终质量验收也使用部署工件及项目真实预后处理。

本机已有的 `05_runtime/builds/vision-export` 是 CPU PyTorch 导出环境，2026-10-20 的包信息为 PyTorch `2.9.1+cpu`、Ultralytics `8.4.175`、ONNX `1.23.2` 和 pycocotools `2.0.11`。GPU 训练环境在独显设备上独立准备，版本与实验一起记录。

## 阶段路线

| 阶段 | 亲自完成的工作 | 主要工具 | 交付结果 |
|---|---|---|---|
| 1 理解模型与数据 | 读清输入输出、预后处理和任务；整理标注数据及划分 | 源码、模型清单、图查看工具、数据检查 | 模型契约、数据清单与质量评测定义 |
| 2 建立双基线 | 测原模型效果、引擎和完整调用；记录资源与关键共存场景 | 质量评测程序、trtexec、阶段计时、tegrastats | 优化前质量和性能报告 |
| 3 定位部署问题 | 判断成本来自预处理、引擎、传输、后处理或等待 | cProfile、nsys；按证据使用 perf 或 ncu | 热点证据与一项可验证的假设 |
| 4 优化部署实现 | 按热点修改一项，例如缓冲复用、向量化、拷贝或执行配置 | 代码修改、对应 profiler、固定 benchmark | 保持模型权重的实现候选与前后对照 |
| 5 实践 FP16 | 构建受支持的 FP16 候选，检查实际执行配置和效果 | TensorRT、质量与性能评测 | 精度和部署成本的第一组取舍 |
| 6 实践 INT8 | 准备校准数据，走通 PTQ、导出、引擎构建和评测 | 所选量化工具、TensorRT、质量评测 | INT8 候选、校准记录、效果与性能对照 |
| 7 实践结构化剪枝 | 分析可剪结构、生成候选、微调、导出与部署 | PyTorch、结构化剪枝工具、训练及评测程序 | 剪枝前后与微调后的完整证据 |
| 8 组合与系统验收 | 筛选候选，组合有效改动，复测质量、共存和持续运行 | 固定整机基准、资源日志、必要的短 trace | 最终候选、取舍说明和可复现报告 |

教学按阶段推进，定位与验证允许往返。每个准备保留的候选完成局部评测后，就测试受影响的共存场景；完整长时验收放在阶段节点。QAT 可在 PTQ 效果不足时开展，蒸馏作为后续扩展。

## 质量与性能双基线

### 数据与质量

准备有标注的公开检测与分割数据，并补充机器人使用场景中的距离、光照、遮挡、运动和多人输入。少量固定图片用于调试与输出核验，任务质量需要独立的标注评测。

训练与微调数据、量化校准数据、开发验证集和最终测试集分别记录用途与重叠情况。最终测试集不参与训练、校准或反复调参。先在开发验证集筛选候选，再对确定的方案执行最终验收。

| 项目 | 记录内容 |
|---|---|
| 人体检测主指标 | person AP50、AP50–95，业务阈值下 Precision 和 Recall |
| 分割与现有任务回归 | 需要保留的 mask 效果及其他在用类别，必要的测距与触发回归 |
| 错误分析 | 漏检、误检、小目标、遮挡、多人等分场景案例 |
| 评测配置 | 数据版本、输入大小、补边、置信度阈值、NMS、最大目标数 |

训练框架验证与生产阈值评测分别记录。模型质量评测可使用 Ultralytics 的验证能力；最终 TensorRT 工件需要接入相同标注评测，并保持项目预后处理与指标定义一致。[Ultralytics 验证说明](https://docs.ultralytics.com/modes/val/)

### 速度与资源

| 范围 | 必须记录的项目 |
|---|---|
| 引擎 | GPU 执行、H2D、D2H、提交开销、延迟分布、吞吐和运行配置 |
| 完整调用 | 图像预处理、同步原生调用、后处理、完整调用的 P50/P95 |
| 资源 | 稳态与峰值内存、最低可用内存、平均和峰值功耗、温度与时钟 |
| 多模型系统 | 相同输入与负载下的视觉 FPS、帧龄、失败/丢弃、对话响应及内存余量 |

原生调用、CUDA 事件区间与 kernel 时间按各自边界解释。文件大小、参数量和 FLOPs 用于说明模型变化，板端实际延迟与资源决定部署收益。

## 独立候选与组合实验

不同方法先保留独立对照，避免把多项变化混在一起：

```text
原始权重与 FP32 实现
  ├─ 部署实现优化 保持权重和精度
  ├─ FP16 保持模型结构与任务
  ├─ INT8 校准后验证最终引擎
  └─ 浮点权重结构化剪枝 微调 导出 在相同精度下比较

独立候选筛选后 → 组合有效改动 → 质量与整机复测
```

剪枝从浮点网络和权重开始，微调后重新导出和构建引擎。剪枝与量化可以进一步组合，但要先知道它们各自的效果。训练中观察到的损失变化，也要落实到部署后的任务指标。

INT8/PTQ 与 QAT 的工具可参考 [NVIDIA Model Optimizer](https://nvidia.github.io/Model-Optimizer/guides/_basic_quantization.html)，结构化剪枝工具可参考 [Torch-Pruning](https://github.com/VainF/Torch-Pruning)。具体操作前在独立环境验证与当前 YOLO 分割网络、Ultralytics 版本、ONNX 及板端 TensorRT 的兼容性。

## 每个候选的验收

一次候选实验按以下顺序收集证据：

1. 记录唯一改动、代码与模型版本、输入和预期作用。
2. 验证输出契约；实现优化检查语义，低精度或剪枝候选检查任务质量变化。
3. 测引擎与完整调用，分别记录速度、内存和功耗。
4. 对照同一基线和验收门槛，分析收益、退化与失败案例。
5. 对准备保留的候选测关键多模型场景，记录对视觉与对话的影响。
6. 保留或回退候选；阶段性组合方案再做完整回归与持续运行测试。

模型更小或参数更少，仍可能没有板端加速。候选没有达到预期时，保留数据并解释原因；学习验收看是否完整完成分析与验证，部署验收看是否满足任务要求。

最终报告至少包含：原始基线、部署实现候选、FP16、INT8、结构化剪枝与微调、组合候选各自的质量、耗时、资源、共存结果及选择理由。表格和记录字段沿用 [优化指南的实验模板](inference-optimization-guide.md#实验记录)。

## 当前进度与下一步

2026-10-20，用户在 Jetson 上完成了原始 FP32 引擎的 trtexec 初测，使用随机输入、预热 1 秒、测量约 10 秒，共记录 748 次推理：

| 项目 | 用户本次初测 |
|---|---:|
| GPU 执行平均值 | 13.3936 ms |
| 含传输延迟平均值 | 13.9936 ms |
| 含传输延迟 P95 | 14.0479 ms |
| 工具调度下吞吐 | 74.5538 qps |

这是引擎层的速度参考。人体 AP、业务阈值下的 Precision/Recall、完整调用、资源及系统共存基线还需要补齐。

已有 B1 测量入口因当前开发源码变化触发冻结校验，已准备匹配协议的 `.worktrees/b1-body-baseline`，单模块协调程序支持 `--source-root`。副本共享模型、环境和固定输入等资产，保留独立的 `runtime-data`；当前项目仍负责管理服务。

2026-10-20 用户确认 B2 模型替换已完成。当前默认服务为 B2，四服务就绪且应用空闲；人体检测仍使用上述相同的 YOLOv8n-seg 引擎。当前源码通过 B2 冻结协议校验，可直接使用 `--protocol benchmarks/manifests/b2-vision-measurements-v1.json` 复现当前单模块基线。B1 独立副本保留作历史复现。

下一轮由用户在项目根目录执行，测量程序会临时停止并恢复当前服务：

```bash
BODY_RUN="05_runtime/validation/body-b2-baseline-$(date +%Y%m%d-%H%M%S)"
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_modules.py \
  --protocol benchmarks/manifests/b2-vision-measurements-v1.json \
  --role detection_segmentation --output "$BODY_RUN"
```

先读取完整调用及预处理、同步原生调用、后处理、资源记录；质量数据与原模型效果基线同步准备，双基线齐备后再选择第一项优化实验。训练与剪枝微调使用用户可用的 NVIDIA 独显电脑或服务器。

## 本机测量目录清理

2026-10-20 按用户要求删除旧测试与临时输出，`05_runtime/validation/` 当前仅保留 `body-b2-baseline-20261020-165328/`。完整调用数据在其 `detection_segmentation/report.json`，引擎初测日志一并保存在 `trtexec.log`。不设置固定报告入口。

仍被验证脚本使用的参考输入和模型来源记录迁入 `05_runtime/datasets/validation-inputs/`，读取脚本兼容原元数据中的路径。版本化汇总与冻结协议继续保留在 `benchmarks/`；已删除的旧原始测量目录不再作为本机可访问路径。

## Nsight Systems 阶段采集

使用单模块脚本的 `--nsys` 选项。协调进程先管理当前服务，再调用官方 `nsys profile` 采集实际模型 worker；实际命令会显示并写入 `detection_segmentation/worker-command.json`。服务控制过程位于采集范围之外。

```bash
TRACE_DIR="05_runtime/validation/body-b2-baseline-20261020-165328/nsys-$(date +%Y%m%d-%H%M%S)"
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_modules.py \
  --protocol benchmarks/manifests/b2-vision-measurements-v1.json \
  --role detection_segmentation --nsys --output "$TRACE_DIR"
```

成功后显示 `detection_segmentation PROFILED`，报告位于 `$TRACE_DIR/detection_segmentation/timeline.nsys-rep`。该模式产生诊断计时，普通基线继续使用不带 `--nsys` 的测量。

NVTX 标记仅在诊断 worker 中启用，范围包括 `frame/输入/调用类型/序号`、`adapter_total`、`preprocess`、`native_sync`、`input_validation`、`runtime_helper`、`postprocess` 和 `mask_decode`。它们描述主机代码区间，需与 CUDA API、传输和 GPU 内核时间线对照。测量结束或异常时恢复原函数。

NVTX Python 包以固定版本和哈希安装在 `05_runtime/builds/profiling-python/`，依赖记录为 `01_src/benchmarks/requirements-profiling.lock`，由采集进程显式加载。

2026-10-20 17:51 的首次采集在停止服务时失败：应用报告退出超时，但所有受管进程组随后已空，服务恢复 READY。该次 `report.json` 的 `modules` 为空，生成的 trace 不包含模型测量。此问题与已有 `B2-EXIT-01` 相符，根因仍待定位。

诊断模式允许在确认所有服务归属正确、主 PID 为零、进程组为空且状态为 inactive/failed 后继续；退出异常保存到 `shutdown-warning.json`。有残留进程或状态不能确认时仍拒绝采集。普通基线保持原有的正常退出要求，退出时限未调整。
