# Orin NX 端侧多模型陪伴服务

在 Orin NX 16GB 上连接 D435i、十模型视觉、中文语音和 4B 图文模型。重点是同帧数据流、板载资源边界、可观察性和可复现测量。性能优化由用户接手逐项完成。

> 整合分支验收进行中：工具与核心模块已完成阶段验证，当前运行包已通过独立恢复，最终性能对照和主分支交付仍待完成。实际进度见 [执行清单](benchmarks/manifests/consolidation-progress.json)，完整范围见 [本轮 Goal](docs/consolidation-goal.md)。

[架构与代码地图](docs/project-overview.md) · [模型与接口](docs/models.md) · [基准与复现](docs/visual-baseline.md) · [依赖与恢复](01_src/deploy/dependencies-layout.md) · [服务管理](01_src/deploy/systemd/README.md)

## 从停止状态启动完整程序

本机项目运行依赖在 `05_runtime/`，普通 Git 克隆不含模型、引擎和 Python 环境。新副本先按恢复说明取得经过验收的匹配运行包，再使用兼容的 Jetson Linux、CUDA、TensorRT 和 Python 3.12 主机环境。

```bash
cd /home/nvidia/Multi_Model_Inference
./run.sh config
./run.sh services check
./run.sh services up
./run.sh services status
```

`run.sh`、直接 Python 入口和服务管理器共享默认配置 `02_config/settings.orin-nx.yaml`。优先级为显式 `--config`、`EDGE_ASSISTANT_CONFIG`、项目默认配置；基础字段来自 `02_config/settings.yaml`。使用 `./run.sh config` 核对实际生效参数，避免环境变量影响判断。

`services up` 启动 `llama`、`asr`、`speech`、`app` 四个服务。看到 `READY` 后，在板端打开 `http://127.0.0.1:8090/`。远程电脑可建立端口转发：

```bash
# 在电脑上运行，将 <板端IP> 换为实际地址。
ssh -L 8090:127.0.0.1:8090 nvidia@<板端IP>
```

随后在电脑浏览器打开同一地址。`/debug` 显示相机预览、实际采集/处理 FPS、帧龄、各模型耗时和板端资源；无需给板子连接显示器。面板是实时采样，正式性能比较使用冻结输入基准。

先确认相机有新鲜画面，再提交一句文字问题，检查回复与生成的语音文件。当前音频默认使用文件输入/输出；没有连接麦克风或扬声器仍可运行软件链路。ReSpeaker Lite 实物录播、AEC 和双讲尚需后续验收。

```bash
./run.sh ask "你好" --json-output
./run.sh services logs --components app --lines 80
./run.sh services down
```

`ask` 直接检查模型接口。完整对话、工具、取消和语音发布通过 Web 应用验证。结束一轮后再停止服务；`B2-EXIT-01` 退出超时仍是开放问题，不能把强制回收记作正常退出。

## 当前能力与边界

- 视觉：检测分割、二维全身/双手姿态、人脸检测/特征、表情、头姿/视线、行人/物体外观特征和轨迹关联。
- 深度：D435i 原生输出及 SDK 对齐；十个视觉网络均使用 TensorRT，精度和模型文件保持既定方案。
- 交互：挥手问候、静止张掌取消整轮；停止生成与输出，阻止未执行工具，保留已提交的笔记或提醒。
- 对话：Qwen3.5-4B、文件语音 ASR/TTS、短会话上下文、笔记/提醒与幂等回执。

身份匹配是可选视觉能力，语音指令没有身份授权关卡。机器人运动、导航、长期记忆和神经双目深度继续后置。软件回放通过不等于真人手势精度、真实声音或长期稳定性已验收。

## 代码怎么读

```text
01_src/
  app/          应用、采集、视觉、轮次控制、语音、动作和诊断
  deploy/       服务归属、原生后端、工件、构建和运行包
  validation/   功能、原生参考、回放、故障与恢复检查
  benchmarks/   冻结输入、单模型、共存与只读观察
  tooling.py    工具任务分发
02_config/      基础字段与唯一当前平台配置
03_models/      当前模型、引擎和辅助组件清单
04_web/         Web 交互与调试界面
05_runtime/     本机运行工件、状态、验证和基准数据；不进入普通 Git
scripts/        build.py、validate.py、benchmark.py、runtime.py 四个薄入口
tests/          软件回归与固定契约样本
benchmarks/     输入协议、测量结果、必要失败证据和整合清单
docs/           当前架构、交接与优化学习材料
licenses/       第三方许可证
references/     用户保留的参考材料与仓库
```

阅读顺序：`main.py` → `app/cli.py` → `app/application.py` → `camera.py` → `vision_pipeline.py` → `vision_models.py`。视觉优化先读 `segmentation.py` 与 `tensorrt_runner.py`；对话控制读 `turn_control.py`、`turn_execution.py` 和 `speech/output.py`。详细所有权和数据流见架构文档。

VS Code Remote SSH 打开项目根目录，远端安装 Python/Pylance，并选择 `05_runtime/environments/edge-assistant/bin/python`。`.vscode/settings.json` 已声明 `01_src/` 搜索路径；若之前手动选过解释器，需要重新选择，再用 F12 或 Ctrl+点击跳转。

## 工具与验证

```bash
source ./05_runtime/environments/edge-assistant/bin/activate
python scripts/build.py --help
python scripts/validate.py --help
python scripts/benchmark.py --help
python scripts/runtime.py --help
python scripts/build.py artifacts --check-only
python -m pytest -q
```

入口只选择任务，公共代码不导入 `scripts/`。构建/导出写候选清单，验证后再选择为运行配置。原生参考需要额外开发环境，见模型说明；恢复运行包不自动安装训练/导出环境。部分验证和基准会停止并恢复本项目服务，运行前查看任务帮助和适用范围。

已完成的阶段证据包括十模型固定输出对照、完整管线回放、文字/文件语音与取消/动作链路，以及工具整合回归。最新计数和证据摘要位于执行清单，最终交付还需同输入性能对照和独立恢复。

## 如何接手优化

先自己完成一次停止、启动、观察、提交请求和关闭，确认理解数据流及资源所有者。随后按 [人体检测优化实操](docs/body-detection-optimization.md) 和 [推理优化学习指南](docs/inference-optimization-guide.md) 建立任务质量与性能基准，用 Nsight Systems/NVTX、trtexec 和应用指标定位瓶颈，再一次只改一个因素并比较。

历史源码和计划从 Git 标签 `before-mainline-consolidation-20261020`、`before-historical-file-retirement-20261021` 恢复；主线不再提供旧版日常启动选项。整合前 B2 的固定输入、引擎、基准和关键失败证据继续保留；用户原始 ONNX、`references/` 和第三方许可证不随旧版本清理删除。
