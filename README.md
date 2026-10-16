# Orin NX 无机身多模态陪伴服务

基于[NVIDIA Reachy Mini Jetson Assistant](https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant)的派生工程：保留其语音、视觉、模型接口、网页和监控，裁掉机器人运动表达，在Orin NX16GB上接入中文语音、4B主模型及人体/人脸/手部感知。

**当前主目录和 `main` 已采用 Reachy 派生实现。** 默认运行四个服务：4B 图文模型、ASR、OpenVoiceStream 语音服务和 Reachy 应用。人体/脸/手、图文问答、中文文件语音、当前会话、笔记/提醒及整轮取消已贯通；真实手势负例/遮挡/多手质量和 B0-SW 尚待完成。

主目录：`/home/nvidia/Multi_Model_Inference`。网页：`http://127.0.0.1:8090`。使用独立环境管理服务：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services status
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services up
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services down
```

操作细节见 [部署与回退](deploy/systemd/README.md)，数据流和阶段记录见 [迁移说明](docs/reachy-migration.md)。当前验证包括 167 项自动化测试、20 次连续完整应用启停、实际进程崩溃恢复、两次旧系统回退、20 轮真实模型共存压力及隔离网络离线启动。证据见 [部署报告](benchmarks/results/reachy-r4-deployment-summary.json)。这些是软件功能与可靠性检查，尚未交付性能基线。基线的 3 份图像、20 份合成中文语音及[测量协议](benchmarks/manifests/reachy-b0-sw-v1.json)已冻结，正式测量待进行。

按用户 2026-10-16 决定，麦克风/扬声器实物环节后置；真实手势补测稍后进行。当前继续软件框架与 B0-SW 交接，交接后由用户先开展视觉优化。旧系统固定在 `~/AI_Stack/archives/legacy-runtime-972de50`，默认旧七服务已停止。

## 当前确定的流程

1. 固定上游源码与模块职责，完成无机身适配并汇入主项目。
2. 接通完整中文语音、视觉、手势与输出，修复生命周期和异常问题。
3. 整体功能跑通后，由助手交付各模块与整机当前性能基线。
4. 用户在逐步指导下，使用专业工具逐模块定位、分析、优化、回归并对比基准。

挥手触发问候；静止举掌保持片刻取消当前整轮，停止生成和播放，阻止未开始的工具；已完成笔记/提醒保留。身份验证删除，长期记忆后置。主模型暂用Qwen3.5-4B，首版后端采用llama.cpp。

## 审查入口

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
