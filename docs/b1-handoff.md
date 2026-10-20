# B1 启动与视觉实操

本页是 B1 的操作和阅读入口。2026-10-20：代码与输入契约已冻结，单模块及共存基线已测量；持续稳定性、运行包独立恢复和默认启动切换仍在验收。当前无参数启动仍为 B0，以下命令显式选择 B1。最终状态以 [验收清单](../benchmarks/manifests/vision-b1-v1.json) 为准。

## 先启动，再阅读

在板端的项目根目录执行。结束当前对话，并确认没有其他基准或部署操作；长时间测试会占用部署锁并在结束后恢复服务。

```bash
./run.sh services down
./run.sh services check --config 02_config/settings.orin-nx-b1.yaml
./run.sh services up --config 02_config/settings.orin-nx-b1.yaml
./run.sh services status
```

`check` 核验模型、原生程序与上游版本；`up` 按依赖顺序启动四个 systemd 服务。`READY` 后打开网页，确认相机持续出图、深度可用、视觉后端为 `tensorrt_b1`，再完成一次文字问答、带图问答和取消。当前没有实体麦克风/扬声器，WAV 输入输出只验收软件链路。

远程电脑可使用 `ssh -N -L 18090:127.0.0.1:8090 nvidia@板子IP`，再打开 `http://127.0.0.1:18090/debug`。实际采集 FPS、视觉处理 FPS、浏览器接收 FPS 是三个指标；模型耗时和结果帧龄也不是同一个指标。调试页面的实时读数帮助定位现象，正式前后对照使用固定基准。

停止使用 `./run.sh services down`；日志使用 `./run.sh services logs --components app --lines 80`。回退 B0 时先停止全部服务，再执行 `./run.sh services up --config 02_config/settings.orin-nx-headless.yaml`。不要让两套应用同时占用相机和模型资源。

## 按数据所有者阅读

| 顺序 | 入口 | 要弄清的问题 |
|---|---|---|
| 1 | `run_web_vision_chat.py`、`01_src/deploy/services.py` | 四个进程谁启动、谁等待就绪、谁退出？ |
| 2 | `01_src/app/camera.py` | RGB 与原生深度如何对齐、标记时间和源标识，最新帧与 JPEG 历史如何限额？ |
| 3 | `01_src/app/perception_b1.py` | 检测后如何按距离与数量选目标，为什么未触发的模型可以没有输出？ |
| 4 | `01_src/app/vision_models.py`、`vision_roi.py`、`wholebody_pose.py` | 每个模型如何加载、准备输入、推理、还原坐标、计数和释放？ |
| 5 | `01_src/app/visual_tracks.py`、`identity_gallery.py` | 跟踪编号如何随相机源失效？人脸标签为何不等于对话授权？ |
| 6 | `01_src/app/gestures.py`、`turn_control.py` | 挥手和静止张掌如何分开，取消如何阻止跨轮次输出与未执行工具？ |
| 7 | `01_src/app/diagnostics.py`、`01_src/benchmarks/vision_b1_stream.py` | 在线指标与正式基准分别测了什么、排除了什么？ |

每帧只保留当前观察所属的源、帧和目标；过期/缺失深度不能冒充有效距离。视觉按串行最新帧处理，每个模型一个上下文，默认最多处理 1 个人物姿态、2 张人脸属性、1 个物体特征目标。增大数量、切换模型或降低工作频率都会改变工作量，比较时必须记录。

## 第一次实验只复现视觉

先阅读 [冻结协议](../benchmarks/manifests/b1-vision-measurements-v1.json) 和 [实测表](visual-baseline.md)，理解 5 张固定图片、均匀模拟深度、预热、统计和触发次数。模型参考核验、性能数据、真人准确率各有不同证据。

```bash
# 真实 TensorRT 连续管线；自动临时停止后端并恢复此前的服务。
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_stream.py \
  --conditions replay-only --output 05_runtime/validation/my-b1-replay-01
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_stream_summary.py \
  --output 05_runtime/validation/my-b1-replay-01
```

输出目录必须是新目录。计时期间关闭额外测试、构建和浏览器负载，保持电源模式与时钟策略。不要修改冻结文件；需要改变实现时先建立新的实验分支、协议与基线编号，保留 B1 原始字节和旧版本。

先解释三件事：处理频率是否接近 4.95 FPS；各输入触发了哪些模型；P95 帧龄为什么高于 P95 整帧处理耗时。随后再测 `replay-conversation`，使用相同视觉输入观察 ASR/4B/TTS 共存下的变化；当前对应值约 3.98 FPS，不能把无人实拍的较低耗时当作优化收益。

## 有证据后选择工具

| 现象 | 先看什么 | 下一步工具与判断 |
|---|---|---|
| 帧龄升高 | 接收到开始、处理到完成、跳过采集帧、失败/过期记录 | 短时间 Nsight Systems 时间线，找 CPU 等待、CUDA 提交和同步的位置 |
| 某模型耗时高 | 同输入的预处理、H2D、引擎执行、D2H、后处理原始分布 | `trtexec` 检查引擎级执行；与包含预后处理的适配器计时分开 |
| 与 4B 共存后变慢 | 同工作量单跑/共存的帧率、帧龄、请求延迟 | Nsight Systems、tegrastats、进程 PSS 联合观察；GPU 忙不能独自证明具体瓶颈 |
| 内存持续增长 | 各进程 PSS/RSS、FD、线程、缓存与目标数 | 分段趋势和重复关闭/重开，区分缓存高水位、采样器开销与未释放资源 |
| 已确定某 CUDA kernel 是热点 | 该 kernel 的调用路径和输入 | 环境支持时用 Nsight Compute；不能用其高开销结果充当正式端到端基线 |

工具版本与权限先按板端能力确认，不为工具升级 BSP/CUDA/TensorRT。下一步由你运行第一组命令、描述观察，我再针对一处问题给出 1～3 条分析命令。每轮只改变一个因素，保存原始数据，复查输出一致性、功能与共存影响；不提前自动完成你的优化实验。

## 已知覆盖边界

已完成参考一致性、固定公开图片张掌触发、真实无人物场景的 RGBD/对话共存，以及受控失效/重连检查。真人挥手、多人遮挡/交叉、实际距离边界和识别精度需要单独补测；早期 B0 的手部关键点记录不能替代 B1 模型的视频验证。视线对小于 4 像素的眼部 ROI 会拒绝推理，此类计数不等于 TensorRT 故障。

私有 70/52 点定义、完整十类手势和 exaggeration 分值没有等价复现声明；实体声学、导航/机身、长期记忆及工业长时验收按既定范围后置。短时基线和本阶段稳定性窗口不代表 8 小时或无人值守工业认证。
