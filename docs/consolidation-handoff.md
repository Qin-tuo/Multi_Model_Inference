# 代码整合交接：2026-10-21

本轮完成当前 B2 主线的代码整合、旧版日常入口退出、运行包恢复和同条件基准对照。十个视觉模型、TensorRT 精度、D435i 原生深度、4B/ASR/TTS 及四服务方案保持不变。后续性能优化由用户接手。

运行与测量源码为 `714290e`；匹配运行包的源码提交为 `590c110`，两者应用、配置、模型清单和工具的源码字节一致。其后的提交记录交接文档、工件锁和实测结果。入口是项目根目录的 [README](../README.md)。

## 已交付什么

| 范围 | 结果 |
|---|---|
| 恢复与其他工作 | 建立 Git 恢复标签，分阶段保存未提交修改；合入主分支时保留 28 个待提交文件，包括用户的 COCO 优化学习计划 |
| 启动与配置 | `run.sh`、Python 入口、服务管理器统一默认配置，显式配置与环境变量覆盖仍有效 |
| 核心代码 | 提取帧处理、模型工件和姿态几何；应用组装、消息发布、轮次状态、单轮执行、语音采集/输出各自负责资源 |
| 工具 | 构建、验证、基准、部署逻辑进入正式模块，`scripts/` 只保留四个任务入口；源码不反向导入脚本 |
| 旧版退出 | 旧版专用业务实现与 137 个历史日常文件退出主线，必要失败记录和原 B2 证据保留 |
| 运行包 | 约 7.68 GB，17,224 项文件/链接；独立副本隐藏原项目及 AI_Stack 路径后，通过相机、文字、文件语音与四进程停止检查 |
| 工件清理 | 移除旧归档、已退役暂存元数据和四个未选用姿态候选文件，约 7.60 GB 逻辑大小；498 项保护工件保留 |
| 功能与界面 | 最终 39 帧完整管线回放通过，十模型均实际调用，标注张掌片段触发停止且未触发挥手；图库重开通过；Firefox 实际渲染十行模型诊断面板 |
| 输出与回归 | 22 组模型输出与原 B2 一致；最终串行回归 302 项通过 |

清理不按 `b1/`、`b2/` 目录名判断。现用七个沿用模型、旧基准工作区的共享依赖、导出/原生参考环境仍有用途。用户上传的八个原始 ONNX、`references/` 和许可证保留。`segmentation-repeat` 的原始 ONNX 与现用导出文件摘要不同，因此作为历史实验保留；没有删除其链接指向的模型。

[执行清单](../benchmarks/manifests/consolidation-progress.json) · [文件处置清单](../benchmarks/manifests/consolidation-inventory.json) · [验证摘要](../benchmarks/evidence/consolidation-verification.json) · [运行包锁](../01_src/deploy/project-runtime.lock.json)

## 当前基线

五组相同固定输入回放的实测完成率：

| 条件 | 原 B2 FPS | 当前 FPS |
|---|---:|---:|
| 仅视觉 | 5.00 | 4.98 |
| 其他模型驻留 | 4.98 | 4.98 |
| 4B 持续推理 | 4.58 | 4.57 |
| 语音对话 | 4.37 | 4.38 |
| 带图问答 | 4.42 | 4.43 |

七组各 60 秒的正式窗口共处理 1,997 帧，整帧失败和采样截止时间遗漏均为 0。固定回放共存负载的最低可用内存约 3.09 GiB。完整的单模型、处理延迟、帧龄、任务调用及资源指标见 [基准说明](visual-baseline.md) 和 [前后对照](../benchmarks/results/consolidation-comparison.json)。

两项较大差异已补充调查：头姿偏慢结果可在旧版重现；仅视觉回放帧龄 P95 的补充旧／新对照约为 441.2／443.8 ms。没有复现这两项明显的版本特有退化。初测和重复测量全部保留，不能把这些短窗口当作统计等价证明。`replay-4b` 初测处理 P95 增加约 4.4%，也如实记录。

此次两个实时相机窗口没有触发人体姿态和人脸属性调用，其 FPS 不能代表全部模型满负载，更不能与原有人场景直接计算加速比。正式优化应先固定输入或场景，核对任务调用次数，再比较结果。

## 仍开放或后置的事项

- **B2-EXIT-01：开放。** 旧主线曾多次出现应用/语音停止超时。当前独立恢复四进程退出码均为 0，本轮当前服务停止也正常，但未定位间歇性问题的根因，不能据此宣布修复。
- **短时限测试波动：保留证据。** 首次最终回归与原生回放并行时，模拟健康检查的 150 ms 时限有一次超时；随后健康检查模块 9 项通过、完整串行回归 302 项通过。未调整时限或运行代码；不能从复跑通过推定首次超时的具体原因。
- **补充测量脚本错误：已处置。** 一次临时协调脚本持有构建独占锁，新版回放被正确拒绝且未执行推理。有效旧版结果保留，新版通过正式基准入口重测成功；失败运行没有计入性能基线。
- 实体麦克风/扬声器、AEC/双讲、真人手势准确率和长时间稳定性仍待验收；长期记忆继续后置。

## 从这里接手

先在项目目录完成一次完整启动和观察：

```bash
./run.sh config
./run.sh services check
./run.sh services up
./run.sh services status
```

浏览器打开板端 `http://127.0.0.1:8090/`，查看相机、提交文字请求，再打开 `/debug` 核对十模型指标和当前基线。远程电脑按 README 建立 SSH 端口转发。看完 [架构与代码地图](project-overview.md)，从 `app/application.py` → `vision_pipeline.py` → `segmentation.py` → `tensorrt_runner.py` 阅读实际调用链。

复现性能与功能时使用新的输出目录，命令会协调本项目服务：

```bash
source ./05_runtime/environments/edge-assistant/bin/activate
python scripts/benchmark.py vision --output 05_runtime/validation/user-modules-01
python scripts/benchmark.py stream --output 05_runtime/validation/user-stream-01
python scripts/benchmark.py stream-summary --output 05_runtime/validation/user-stream-01
python scripts/validate.py pipeline --gesture-fixture benchmarks/manifests/b2-held-palm-v1.json --output 05_runtime/validation/user-pipeline-01
python -m pytest -q
```

测量、原生回放和短时限单元测试分开运行。当前协议已经冻结，不要覆盖旧结果或修改哈希来绕过检查。首次优化仍从 [人体检测实操](body-detection-optimization.md) 和 [YOLOv8n-seg + COCO 练习](yolov8n-seg-coco-optimization.md) 开始：由用户建立质量基准、采集 Nsight/NVTX、定位瓶颈，再修改一个因素并与本基线比较。本轮没有替用户实施这些优化。

原始新测量位于 `05_runtime/validation/consolidated-*-20261021/`，交接审计位于 `05_runtime/refactor-checkpoints/consolidation-20261021-final/`；这些本机文件不进入普通 Git。Git 保存汇总、协议、摘要和处置记录；运行包补充模型、环境和固定复现输入。独立恢复方法见 [依赖说明](../01_src/deploy/dependencies-layout.md)。

源码恢复点为 `before-mainline-consolidation-20261020` 和 `before-historical-file-retirement-20261021`。旧运行归档已被经过独立验证的新包替代，清理前摘要与原暂存元数据保存在交接审计目录中。
