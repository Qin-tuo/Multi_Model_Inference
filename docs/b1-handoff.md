# B1 启动与视觉实操

本页是 B1 的操作和阅读入口。2026-10-20：`run.sh` 默认启动 B1 R2，四服务、原生深度、文字→4B→TTS 和浏览器相机/调试页面已实际验证。10 个模块和七组整链路基线已测量，R2 运行包通过独立副本离线恢复。当前尚未关闭实时挥手漏触发，整体功能验收仍未完成；最终状态以 [验收清单](../benchmarks/manifests/vision-b1-v1.json) 为准。

原 R1 完成了 20 分钟固定回放及 10 分钟实拍共存窗口。R2 只修复手势时序规则，模型与参数未变，整链路七组窗口和运行包已更新。当前基准命令默认选择 v2 协议；旧协议在修改后的代码上会拒绝运行。复现历史 R1 时使用源码恢复点 `before-gesture-wave-fix-20261020`；原始工件保持保留，R1 持续窗口不冒充 R2 重测。

## 先启动，再阅读

在板端的项目根目录执行。结束当前对话，并确认没有其他基准或部署操作；长时间测试会占用部署锁并在结束后恢复服务。

```bash
./run.sh services down
./run.sh services check
./run.sh services up
./run.sh services status
```

启动器尊重已有 `EDGE_ASSISTANT_CONFIG` 和显式 `--config`。直接执行 `python main.py` 时请加 `--config 02_config/settings.orin-nx-b1.yaml`，避免落到历史 B0 默认值。

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

先阅读 [当前 R2 冻结协议](../benchmarks/manifests/b1-vision-measurements-v2.json) 和 [实测表](visual-baseline.md)，理解 5 张固定图片、均匀模拟深度、预热、统计和触发次数。模型参考核验、性能数据、真人准确率各有不同证据。

```bash
# 真实 TensorRT 连续管线；自动临时停止后端并恢复此前的服务。
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_stream.py \
  --conditions replay-only --output 05_runtime/validation/my-b1-replay-01
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_b1_stream_summary.py \
  --output 05_runtime/validation/my-b1-replay-01
```

输出目录必须是新目录。计时期间关闭额外测试、构建和浏览器负载，保持电源模式与时钟策略。不要修改冻结文件；需要改变实现时先建立新的实验分支、协议与基线编号，保留 B1 原始字节和旧版本。

先解释三件事：处理频率是否接近 4.97 FPS；各输入触发了哪些模型；P95 帧龄为什么高于 P95 整帧处理耗时。随后再测 `replay-conversation`，使用相同视觉输入观察 ASR/4B/TTS 共存下的变化；当前 R2 对应值约 3.93 FPS，不能把无人实拍的较低耗时当作优化收益。

## 有证据后选择工具

| 现象 | 先看什么 | 下一步工具与判断 |
|---|---|---|
| 帧龄升高 | 接收到开始、处理到完成、跳过采集帧、失败/过期记录 | 短时间 Nsight Systems 时间线，找 CPU 等待、CUDA 提交和同步的位置 |
| 某模型耗时高 | 同输入的预处理、H2D、引擎执行、D2H、后处理原始分布 | `trtexec` 检查引擎级执行；与包含预后处理的适配器计时分开 |
| 与 4B 共存后变慢 | 同工作量单跑/共存的帧率、帧龄、请求延迟 | Nsight Systems、tegrastats、进程 PSS 联合观察；GPU 忙不能独自证明具体瓶颈 |
| 内存持续增长 | 各进程 PSS/RSS、FD、线程、缓存与目标数 | 分段趋势和重复关闭/重开，区分缓存高水位、采样器开销与未释放资源 |
| 已确定某 CUDA kernel 是热点 | 该 kernel 的调用路径和输入 | 环境支持时用 Nsight Compute；不能用其高开销结果充当正式端到端基线 |

本机已确认 `nsys` 2025.6.3.541、`ncu` 2026.1.1.0 可执行；`nsys status --environment` 的进程树 CPU 采样检查通过，当前普通用户的系统级采样检查未通过（`perf_event_paranoid=2`）。工具可用不等于已采集或分析了热点；不为工具升级 BSP/CUDA/TensorRT 或修改主机权限。下一步由你运行第一组命令、描述观察，我再针对一处问题给出 1～3 条分析命令。每轮只改变一个因素，保存原始数据，复查输出一致性、功能与共存影响；不提前自动完成你的优化实验。

## 已知覆盖边界

已完成参考一致性、公开图片张掌触发并取消真实生成轮次、实机 RGBD/对话共存，以及受控失效/重连检查。R2 实拍基线中每个完成帧均有模型预测的人体/人脸和姿态任务，但未做真值标注，不能据此声明精度。

[动态回放记录](../benchmarks/results/b1-motion-summary.json) 保留了三段演示未触发的结果；[R2 修复记录](../benchmarks/results/b1-wave-fix-summary.json) 的固定选帧模式已完成真实 TensorRT→问候→4B→TTS，普通实时模式仍漏触发。固定选帧会等待处理、累积输入延迟，不能代替实时验收。真人挥手、多人遮挡/交叉、实际距离边界和识别精度仍需补测；早期 B0 关键点记录不能替代 B1 模型的视频验证。视线对小于 4 像素的眼部 ROI 会拒绝推理，此类计数不等于 TensorRT 故障。

私有 70/52 点定义、完整十类手势和 exaggeration 分值没有等价复现声明；实体声学、导航/机身、长期记忆及工业长时验收按既定范围后置。短时基线和本阶段稳定性窗口不代表 8 小时或无人值守工业认证。

## 真人手势补测

先在 `/debug` 确认人体、左右手和距离有效，手腕与五指完整入镜。以下命令只观察已运行的应用；不会重启服务或修改模型。每次使用新的输出文件：

```bash
./05_runtime/environments/edge-assistant/bin/python scripts/probe_gestures.py \
  --seconds 20 --save-preview \
  --output 05_runtime/validation/my-b1-wave-01.json
```

记录开始后先收手片刻，随后自然左右挥手，再收手等待问候完成。需要同时核对画面中的实际动作、手势事件、`greeting_started` 和对应轮次的完整文件语音回执；出现一个事件不能单独证明动作识别正确。

静止张掌取消单独记录：给上述命令增加 `--prime-stop-test`，输出改为新文件。该选项在新举掌时提交数数请求，随后保持张掌约 1 秒；检查取消对应的 `target_turn_id`、最终 `cancelled`、资源释放和音频未发布。它会发送真实请求，因此先结束其他对话。负样本、遮挡、多手、距离边界分开记录实际动作和结果，不能以一次正样本替代。

默认只保存模型观察和事件；`--save-preview` 才将 JPEG 保存到同名 `.frames/` 目录，最多 64 MiB/3,000 帧，达到限额会记录拒绝数。图像与模型结果只按相同 `source_epoch` 和 `frame_id` 配对，不能按文件序号或最近时间猜配。JPEG 是有损预览，不含原始深度像素，不能作为精确 RGBD 重放或 TensorRT 数值一致性输入。样本和图像留在本地忽略目录，提交前只提取已审查的统计。
