# Orin NX 无机身多模态陪伴服务

基于[NVIDIA Reachy Mini Jetson Assistant](https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant)的计划中派生工程：保留其语音、视觉、模型接口、网页和监控，裁掉机器人运动表达，在Orin NX16GB上接入中文语音、4B主模型及人体/人脸/手部感知。

**当前状态：新版Goal已启动，隔离分支已接通无机身网页、D435i、人脸、4B问答及中文文件语音；人体/手部软件接入已验证。网页整轮取消、后端释放确认及下一轮恢复已实测；真人挥手问候与静止举掌取消已取得正例；手势负例/多人矩阵、实体录播及物理停播仍待验证。默认运行系统尚未切换。** 当前仓库app仍为旧Pipecat/ROS2实现。上游只读审查副本位于/home/nvidia/AI_Stack/research/reachy-mini-jetson-assistant-2673202。

## 当前确定的流程

1. 固定上游源码与模块职责，在隔离迁移分支做无机身适配。
2. 接通完整中文语音、视觉、手势与输出，修复生命周期和异常问题。
3. 整体功能跑通后，由助手交付各模块与整机当前性能基线。
4. 用户在逐步指导下，使用专业工具逐模块定位、分析、优化、回归并对比基准。

挥手触发问候；静止举掌保持片刻取消当前整轮，停止生成和播放，阻止未开始的工具；已完成笔记/提醒保留。身份验证删除，长期记忆后置。主模型暂用Qwen3.5-4B，首版后端采用llama.cpp。

## 实施与审查入口

当前实现位于[migration/reachy-headless分支](https://github.com/Qin-tuo/Multi_Model_Inference/tree/migration/reachy-headless)。[隔离迁移说明](docs/reachy-migration.md)包含可运行命令、验证结果和未完成边界。

- [完整新版计划书与最终Goal](02-jetson-multimodel-inference.md)
- [总览与工作分工](00-project-overview.md)
- [上游版本与源码审查锁](deploy/reachy-upstream.lock.json)
- [实施阶段和验收清单](benchmarks/manifests/reachy-headless-roadmap-v3.json)
- [迁移缺口与风险范围](benchmarks/manifests/reachy-gap-ledger-v3.json)
- [端侧部署岗位资料](benchmarks/manifests/target-jd-20261014.json)

## 现有系统与资料

- [旧实现使用说明及历史结果](docs/archive/README-pre-reachy-20261015.md)
- [旧工程进度和证据](benchmarks/manifests/progress.json)
- [外部依赖目录](deploy/dependencies-layout.md)
- [4B默认模型清单](models/manifest.json)

旧系统的模型、数据、代码历史及references/均保留。性能和正确性证据只适用于各自配置，新派生工程必须重新验证。迁移阶段的独立环境、功能证据和进度见任务清单；后续性能优化由用户在指导下完成。
