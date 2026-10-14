# Jetson 本地多模态助手

在 Jetson Orin NX 16GB 上构建具有中文语音、持续视觉、图文理解、长期记忆、打断、工具和提醒能力的本地助手。整体基底采用 Pipecat，记忆采用 Mem0，提醒采用 APScheduler；业务代码负责模型接入、策略和性能测量。实施任务与验收见 [项目计划](02-jetson-multimodel-inference.md)。

## 当前进度

项目处于框架接入与功能闭环验证阶段。框架/插件236项测试、真实GPU文本/图文推理、BGE转换回归、10项语义召回和3项跨应用进程记忆问答已通过。D435i双路640×480采集实测接近30Hz；ROS检测和姿态已在本机TensorRT执行，原生ORT逐节点trace确认完整模型计算位于TensorRT。持续会话已接入实时相机和语音文件输入，并复用记忆、工具和提醒；图文计数与实际推理用量核对通过。完整模型质量、物理语音、响应时延和整机验收仍待完成。逐项进度见 [progress.json](benchmarks/manifests/progress.json)，测试范围见 [测试报告](benchmarks/results/framework-tests.json)。

视觉质量进入[完整COCO2017公开验证集回归](benchmarks/manifests/vision-quality-v1.json)阶段：[5,000张图片与标注已封存](benchmarks/manifests/coco2017-quality-v1.json)，检测的[完整5,000张对照](benchmarks/results/coco-detection-deployment-baseline.json)已完成：bbox AP从26.4618到26.4240，下降0.0378个百分点，满足1个百分点门槛；姿态部分仍在运行。参考为锁定的PyTorch FP32权重，候选直接调用已安装的yolo_ros原生库和现有TensorRT engine；不另写预处理/NMS。保持640输入、置信度0.5、NMS IoU0.45和100个结果上限，以pycocotools评价bbox/OKS AP。姿态主参考保留全部关键点，另列相同发布过滤后的次要参考，不能用次要口径掩盖质量损失。[单图接线报告](benchmarks/results/vision-quality-plumbing.json)确认完整TensorRT节点执行，但检测框最小IoU为0.9737，未过0.98接线对齐门槛；这一失败保留，继续测完整标注集。[输入隔离诊断](benchmarks/results/vision-input-diagnostic.json)进一步确认：该图原始解码相同，原生OpenCV4.8与参考OpenCV5缩放后有185,802个通道值相差1；给同一个PyTorch模型/NMS换入原生输入后，框与原生CPU结果IoU超过0.999999。这定位了当前单图差异，尚不代表全量影响，也没有修改正在评测的代码或参考口径。公开回归不能替代计划要求的独立真机场次、跟踪/手势或视觉问答验收。

**JT-02多模型竞争专项验收已通过**，见[逐条审计](benchmarks/results/jt02-multimodel-audit.json)。[当前模型/engine清单](benchmarks/results/model-engine-inventory.json)登记7类功能模型（TTS包含Matcha与Vocos）、6个在用TensorRT engine及8个未启用版本；逐项记录路径、SHA256、配置绑定、运行证据及已加载原生库。[同一次原生追踪](benchmarks/results/gpu-concurrency-graph-trace.json)覆盖D435i检测、姿态和6次文本/实时图文请求：188,944条独立kernel记录、177条CUDA Graph整体记录，189,121条设备活动全部关联到成功的启动调用。反向检查中，177次图启动全部有执行记录，6,674次仅捕获调用均位于8个成对捕获区间，没有无法解释的遗漏。

在该次插桩诊断中，检测/姿态启动API返回后的排队P95由视觉阶段的14.34/13.32ms，变为VLM请求阶段的20.69/19.85ms；这是同次追踪的阶段观察，不作为因果收益或性能达标结论。26.74秒采样的检测/姿态约11.67/11.78Hz，最新结果年龄P95为179.87/184.83ms。三模型进程正常退出，七个原服务恢复READY。图整体区间包含内部空隙，不等于kernel活跃时间或SM指令同时执行；物理语音、问答质量、JT-03/04/06/07/08及整机验收仍未完成。

节点级追踪的GraphId/GraphNodeId错误在[2026版本](benchmarks/results/gpu-concurrency-trace.json)和[2025版本](benchmarks/results/gpu-concurrency-nsys2025.json)中均保留为FAIL；[最小CUDA更新实验](benchmarks/results/nsys-graph-update.json)未复现该错误，不能声称已修复底层原因。已验证的整体图采集避开该关联路径，同时保留图外kernel。默认采集使用`--graph-trace graph`，`--graph-trace node`保留用于诊断。原始文件位于`/home/nvidia/AI_Stack/runtime-data/benchmarks/gpu-concurrency-graph-v1/`，Git报告登记SHA256；不保存摄像头图像或公开场景回答。最初的[子进程回收失败](benchmarks/results/gpu-concurrency-cleanup-baseline.json)也保留，回收修复、区间统计及缺失事件识别的[3项专项测试](benchmarks/results/gpu-trace-tests.json)通过。

复现时保持后台空闲并使用新路径：`.venv/bin/python tools/gpu_concurrency_probe.py --run-dir /home/nvidia/AI_Stack/runtime-data/benchmarks/gpu-concurrency-NEW --output benchmarks/results/gpu-concurrency-NEW.json`。脚本临时停止并恢复项目管理的llama、vision、pose，其余服务继续运行。只读清单可通过`.venv/bin/python tools/model_inventory.py --output benchmarks/results/model-inventory-NEW.json`生成；随后运行`python3 tools/gpu_trace_audit.py --trace <追踪报告> --inventory <清单报告> --output <新审计报告>`核对JT-02。旧原始数据、失败报告和模型配置均保留。

最新[冷启动阶段分析](benchmarks/results/voice-cold-stage-diagnostic.json)确认首轮记忆问答的主要等待在模型处理提示：1146个输入token，原生提示计算约1.93秒，模型首token约2.17秒；该轮文件首PCM约4.35秒。已用固定版本的`llama-bench`完成[三种微批量、三个输入长度、各五次测量](benchmarks/results/llama-prefill-screening.json)，没有发现值得采用的收益，当前512微批量保持不变。筛选末尾一次姿态就绪探测未取得新样本，原报告保留FAIL；[后续复查](benchmarks/results/llama-prefill-readiness-recheck.json)七个后端全部READY。原生基准不包含图像编码或完整语音链路，也不替代JT验收。

原生`ngram-simple`推测解码已作为默认关闭的配置候选。[两组各20轮语音/记忆/视觉对照](benchmarks/results/speculative-voice-comparison.json)中，时间问答首文件PCM P95从3.224降到2.297秒，算术从2.003降到1.746秒；冷启动记忆问答仍需4.098秒，候选记忆写入P95为514ms，未满足500ms门槛，因此没有切换默认。两组均无执行错误，检测和姿态时效通过；原配置已恢复，七个后端全部就绪。详细范围与失败项见[评估说明](deploy/evaluation/README.md#原生提示匹配推测解码候选)。

记忆编码现采用 **4个ORT内部线程**，并修复了`memory.embedding_threads`配置未生效的问题。[新一轮2/4线程完整对照](benchmarks/results/embedding-voice-comparison.json)中，写入P95从484降到357ms，查询从159降到140ms，视觉频率/时效通过；编码器60组向量逐元素一致。[公开语音保存与回忆](benchmarks/results/embedding-threads4-public-voice.json)也通过。冷启动记忆问答仍约4.37秒，完整时延和长期验收继续推进。记忆开发集的词级F1及参考答案子串命中不能直接当作语义正确率或正确来源可见率，已有[来源级复核](benchmarks/results/lcvmem-reader-error-diagnostic.json)区分同义表达、证据被预算丢弃和模型漏读。

单来源`flat_v2`格式完成[82题完整对照](benchmarks/results/lcvmem-evidence-format-comparison.json)：同一检索顺序下，18题多保留一条来源，词级F1从28.97到30.58；但也出现具体答案退为泛泛回答的案例，因此[继续作为可选项](benchmarks/results/evidence-format-decision.json)，当前仍用`nested_v1`。[Qwen3.5-4B候选](models/candidates/qwen3.5-4b.json)已完成加载和82题对照：词级F1为31.35，单题墙钟更长且仍有来源理解问题，因此[未选为默认](benchmarks/results/qwen35-reader-decision.json)。当前运行模型仍由[主manifest](models/manifest.json)指定为Qwen2.5-VL-3B。

工具路由已修复“记忆心理学”等普通问题误触记忆清单、查询结果重复积累，以及笔记列表漏标题的问题。3B与4B均通过专项回归；[最新100条工具开发集](benchmarks/results/tool-quality-authoritative-inventory.json)100/100通过，无意外写入或重放新增。此前两轮99/100及输出过滤诊断保留在[评估记录](deploy/evaluation/README.md#记忆主题误判与查询续轮修复)，独立质量和物理验收继续进行。

| 功能 | 组件 | 当前状态 |
|---|---|---|
| 主框架 | Pipecat 1.12.0 原生 PipelineWorker/WorkerRunner | 持续文本/相机/语音文件会话、记忆与工具通过初测；真实设备、完整打断和质量验证继续推进 |
| 文本/图文模型 | llama.cpp CUDA + Qwen2.5-VL-3B Q4_K_M | 37/37层GPU offload及CUDA图像编码；两项问答sanity通过 |
| 长期记忆 | Mem0 2.2.1 + 本机Qdrant 1.19.2服务 + SQLite历史/journal | BGE CPU ONNX、重启召回、更正/遗忘问答通过；完整语义质量集待验证 |
| 提醒 | APScheduler 3.11.3 + SQLAlchemy | 到期文字通知、静音、过期、重启去重通过；语音创建与真实TTS到期文件输出通过，实机播报待验证 |
| 时间语义 | JioNLP 1.5.29 + Python datetime | 按用户原文和请求时间解析中文日期/时长；缺少具体时间时先询问 |
| 语音 | OVS + llama.cpp Qwen3-ASR 0.6B + TensorRT Matcha/Vocos | 连续六轮文件语音打通记忆、工具、图像和提醒；设备/AEC及完整质量、时延待验证 |
| 视觉 | yolo_ros/TensorRT + RealSense/ROS 2 | GPU ORT及ROS五个包编译完成；真实相机检测/姿态运行与逐节点TensorRT证据通过，约12.6～12.8Hz；完整场景质量待验证，FP16失败样例保留 |

Qwen3B作为首轮功能与容量基线。模型身份和产物放在 [manifest](models/manifest.json)，框架适配通过配置读取；更换模型后重新验证质量、上下文预算和共存容量。Edge-LLM留到后续性能对比。

当前包括Qdrant在内的七个后端已同时[就绪](benchmarks/results/full-backend-readiness.json)。以下是此前六个模型/视觉后端的共存记录。持续视觉运行期间，5次[文件语音问答](benchmarks/results/concurrent-file-speech.json)全部识别、回答并回读正确，采样最低可用内存约3.38GiB；同期[40秒视觉测量](benchmarks/results/vision-with-file-speech.json)中，检测/姿态最新结果年龄P95约157/163ms。测量包含后续ASR回读请求；当前镜头没有检出目标，不能据此判断目标/关键点准确率。这是短时共存初测，物理播放、完整负载容量和GPU任务重叠trace仍待验证。

## 使用现有入口

本机外部框架、模型、构建产物与 Python 环境统一放在 **`/home/nvidia/AI_Stack/`**。其中 `frameworks/llama.cpp/` 是 llama.cpp 源码，`builds/llama/bin/llama-server` 是服务程序。目录与版本入口见 [依赖目录说明](deploy/dependencies-layout.md)；本机总目录内还有 `README.md` 和完整 Python 依赖清单 `dependencies.json`。原缓存路径及项目 `.venv*` 保留为兼容链接。

### 持续文本与相机交互

先运行本地模型服务，再在另一终端开启会话：

GPU版ORT已完成构建，模型服务已恢复。启动命令会复用已就绪的受管理服务。构建与驻留模型曾叠加触发OOM，现通过互斥租约阻止重型ORT编译和模型服务同时运行。

```bash
python3 tools/project.py up --components qdrant,llama
# 另一个终端：
python3 tools/project.py interact --user owner
```

相机问答先运行 `python3 tools/project.py up --components camera`，然后在会话内输入 `/look`、`/look 这个物体是什么颜色？` 或“你现在看到了什么？”。`interact`默认订阅ROS对齐RGB-D；纯文本环境使用 `--no-vision`，视觉问题会明确提示不可用。相机订阅本身不会启动相机服务。

可直接输入“请记住，我喜欢茉莉花茶。”“我喜欢喝什么茶？”“创建笔记，标题是采购，内容是周末买绿茶和杯子。”“请查询我的笔记。”或“五分钟后提醒我休息。”。`/new` 开启新会话，`/stop` 停止当前回答，`/memories`、`/notes`、`/reminders`、`/history` 查询实际记录，`/quit` 退出。`--user guest` 使用独立档案；`--session-id ID` 恢复指定档案已输出的会话上下文；`--input-file PATH` 顺序执行最多64KiB的文本脚本。

此入口复用一个持续运行的 Pipecat worker 和原生工具 continuation。模型请求及工具回传请求均通过 llama.cpp 原生 `/v1/chat/completions/input_tokens` 检查实际模板、文字和图像预处理token；优先移除旧轮次，当前问题过长则明确失败。[固定版本接口说明](https://github.com/ggml-org/llama.cpp/blob/08ffba21e5c2f1728b6b405c05ff3913f65706d4/tools/server/README.md#post-v1chatcompletionsinput_tokens-token-counting)。默认会话文本保留7天，仅成功输出的完整文本进入已完成历史；更正/遗忘会清除相关来源和派生会话记录，并失效当前旧上下文。笔记和偏好分别从各自存储读取，购物笔记不作为偏好证据。用户明确操作最多修复重试一次，未实际成功时不会口头确认。

图像请求使用原生 `UserImageRequestFrame` / `UserImageRawFrame`。选择距离提问时间≤150ms的640×480帧，经Pipecat原生JPEG编码（Pillow默认quality75，不缩放）附在本轮消息中；首次提交模型前检查2秒保留期限，生成期间发生相机epoch变化或断流则拒绝消费旧答案。生成耗时可以超过帧缓存期限，同一轮图像始终绑定原始提问时间。完成或取消后释放图像引用并清除上下文中的图像数据URL，默认只持久化已输出文本。

[真实图文会话初测](benchmarks/results/visual-session-smoke.json)覆盖三种已知颜色、图文后记忆召回、实时D435i输入、取消后恢复和断流拒绝；图文计数与实际模型usage一致。实时画面的目标准确率没有地面真值验证，完整JT质量集仍待完成。

当前取消验证覆盖原生 interruption 到达管线末端的确认、HTTP生成请求停止、服务slot空闲及下一轮恢复。取消期间阻止新请求准入；视觉初测从取消到确认模型空闲约2.60秒，不代表声学停播时延。控制台按完整回答输出，尚未实现逐token显示；此验证不证明物理停播、AEC或GPU内存释放。到期提醒已在控制台按“通知：内容”显示，等待当前回答结束再输出。完整语音助手验收仍未完成。

提醒数据库迁到新目录后，启动恢复会重建指向当前数据库的任务参数，避免到期任务写回旧目录；原始失败见[迁移基线](benchmarks/results/reminder-relocation-baseline.json)。恢复工具使用当前通知回执，并由APScheduler重建有效任务，已完成或已取消的提醒不会从旧备份重新排队；结果不确定的播报不自动重试。

提醒静音使用 `/mute-reminders on`、`/mute-reminders off`，设置按档案持久保存。未开始输出的失败最多重试3次，默认间隔5秒；提醒到期后默认有效1小时，可在`configs/runtime.yaml`配置。已开始输出但未确认的提醒保留为`uncertain`，重启后不自动重播，可查询后取消或重新设置。文字显示与音频播放分开记录。

模型服务不可用时，可独立运行文字提醒消费者：

```bash
python3 tools/project.py notifications --user owner
```

同一数据目录只运行一个消费者；`interact`已经包含通知消费。独立模式不需要加载语言模型。跨进程静音、过期、取消、主人/访客隔离及完成后不重播见[文字通知报告](benchmarks/results/reminder-delivery-smoke.json)。

音频通知接入采用 Pipecat 原生 `TTSSpeakFrame` 和有序输出标记，记录实际输出尝试，等待输出设备排空后才确认；合成结束、普通停止讲话事件或部分音频写出都不能作为成功依据。通知已接入持续语音会话并通过真实TTS到WAV验证。LocalAudio适配提供PortAudio排空接口，但麦克风/扬声器未连接，设备排空和声学验收仍待完成。[PortAudio完成语义](https://portaudio.com/docs/v19-doxydocs/portaudio_8h.html)

实际测试见[持续会话报告](benchmarks/results/session-smoke.json)、[独立进程与取消报告](benchmarks/results/application-smoke.json)。这些是合成开发样例；模型曾漏调用工具或把购物内容误当偏好，失败样例与校验修正已保留，不能以这几条通过替代完整工具质量集。

### 持续语音入口

`voice` 使用Pipecat原生语音worker负责Silero、STT、TTS和输出队列，最终转写交给同一个持续助手会话处理。回答和到期提醒共用输出队列；回答在完整写出并完成排空确认后才进入会话记录。文件输出记录为`fixture`，设备输出记录为`audio`。输入文件每份应只有一句完整发言，PCM16、16kHz、单声道、最长30秒。

本机可执行的文件模式：

```bash
python3 tools/project.py up --components qdrant,llama,asr,speech
python3 tools/project.py voice --no-vision --user owner \
  --input-wav "$HOME/AI_Stack/runtime-data/fixtures/roundtrip/request-0.wav" \
  --output-wav "$HOME/AI_Stack/runtime-data/voice-demo.wav"
```

`--input-wav`接受多份文件，按顺序交给同一会话；处理完后退出。输出文件必须使用新路径。可选`--report PATH`保存包含转写和诊断字段的本地报告。启用视觉时去掉`--no-vision`并先启动`camera`，语音“这是什么？”、“画面主要是什么颜色？”会选取与话音结束时间关联的图像；无法取得有效图像时返回错误。

真实语音模型的[六轮集成报告](benchmarks/results/voice-assistant-smoke.json)覆盖记忆写入/召回、时间工具、已知蓝色图像、提醒创建与到期文件输出；另有[公开命令报告](benchmarks/results/voice-console-smoke.json)和[语音文件→实时D435i→回答音频报告](benchmarks/results/voice-live-camera-smoke.json)。前者视觉输入为合成夹具，后者没有场景地面真值。最新六轮文件初测中，从推定话音结束到首段PCM：算术/记忆查询约1.7～1.9秒，时间查询2.48秒，图像问答3.92秒，提醒4.61秒，均未触发工具修复。此前提醒轮14.1秒的[基线](benchmarks/results/voice-before-tool-prefill.json)保留；这些个例尚未通过JT-04完整时延验收。

ASR复用上游分段任务，并增加一个待处理段的队列上限、PCM缓存上限和12秒请求期限。连续发言中的第一段结果不能在后续分段尚未完成时触发模型；取消会使旧输入、迟到转写和旧错误失效。溢出会使整轮失败，不截断后继续执行。[真实ASR延迟回调验证](benchmarks/results/voice-asr-lifecycle-smoke.json)覆盖分段完整性、取消后无记忆副作用及下一轮恢复；它不代表物理打断或完整故障矩阵验收。

设备模式使用 `python3 tools/project.py voice --input-device N --output-device M`，需要PyAudio及明确的PortAudio设备索引，详见[语音部署](deploy/speech/README.md)。ReSpeaker Lite和扬声器未连接，设备录播、AEC、真实双讲以及100次打断仍待验证；当前不能宣称完整实时语音助手已经验收。

最近的语音资源验证修复了重复停止事件遗留STT定时器，以及LLM会话退出后未关闭SDK连接池的问题。[20次真实链路测试](benchmarks/results/voice-resource-lifecycle.json)覆盖取消后恢复对话、等待识别回调时关闭，每类10次，关闭后均无应用任务遗留，ASR/TTS/LLM/计数连接和输出文件均关闭。使用合成音频，并在真实识别完成后暂缓回调；物理停播、GPU抢占和8h稳定性仍待验收。复现与失败基线见[语音部署说明](deploy/speech/README.md#取消与关闭的资源检查)。

另有[50次后端执行中测试](benchmarks/results/backend-cancellation-summary.json)：LLM取消/应用超时、ASR取消/HTTP读取超时、TTS流中断各10次通过，均观察活动执行状态、服务释放及后续请求恢复。ASR/TTS在客户端退出后仍会继续处理，不能据此前端状态认定后端已经空闲。LLM应用取消确认现已拒绝空或格式错误的执行槽状态，避免错误恢复准入。ASR/TTS部分直接测试OVS接口，完整语音应用故障矩阵、GPU逐kernel轨迹及物理打断仍待验收。

[完整语音应用验证](benchmarks/results/voice-inflight-summary.json)已补充40次分阶段取消：ASR、LLM、TTS合成与WAV播报各10次通过，取消后直接提交新语音，没有迟到PCM、旧完成历史或记忆副作用，恢复音频可正确回读。随后修复了SDK隐式重试及启动失败时客户端未关闭的问题；真实终止ASR/LLM/TTS工作进程后的首轮3例均自动重启并恢复同一语音应用，30例重复验证也已通过，均确认旧进程退出且未使用探针补救启动。单字数字回读不稳定、实机声学和完整整机验收继续保留为缺口。

### 长期记忆服务

公开`interact`、`voice`入口使用Mem0和本机Qdrant 1.19.2服务，BGE及SQLite history/journal保持原有职责。先运行`python3 tools/project.py up --components qdrant,llama`；语音还需`asr,speech`，视觉还需`camera,vision,pose`，也可直接`up --components all`。

服务只监听127.0.0.1:6333，认证凭据由启动工具自动生成并保存在运行目录的私有文件中，无需手工配置。`status --components qdrant`核对实际进程、监听socket、认证访问成功及无凭据访问被拒绝。systemd负责重启和进程组清理，日志按既有限额轮转。当前仍需手动`up`，没有启用开机自启动。

每个数据目录拥有独立collection，绑定服务实例、目录和embedding文件指纹。绑定同时记录在本地`store.json`和Qdrant原生collection元数据中；复制目录、连接错误实例、替换embedding或丢失collection会明确失败，避免悄悄打开空库。已有绑定的数据目录沿用自身绑定。旧嵌入式数据通过`python3 tools/project.py migrate-memory --data-dir /实际应用数据目录`显式迁移；先退出使用该目录的应用，Qdrant服务保持运行。工具保留ID/版本/来源，合并删除记录，核对当前BGE向量后分批导入；中断后重复原命令续跑。未完成的迁移可加`--abort`撤销，已完成迁移禁止回退到旧库。详见[迁移说明](deploy/memory/README.md)。

验证包括[绑定后的实际Mem0读写](benchmarks/results/managed-memory-smoke.json)、[公开文字入口](benchmarks/results/managed-memory-console-smoke.json)、[语音文件入口](benchmarks/results/managed-memory-voice-smoke.json)及[真实BGE迁移](benchmarks/results/memory-migration-smoke.json)。文件语音验证了真实ASR/LLM/TTS及记忆写读；没有声学播放回执。配置及复现说明见[记忆部署](deploy/memory/README.md)。

### 应用数据备份

退出使用目标数据目录的应用后，可执行`python3 tools/project.py backup-data --data-dir /应用数据目录 --output /独立备份目录`，再用`python3 tools/project.py verify-backup --backup /独立备份目录`校验。备份包含Qdrant原生快照及记忆、会话/笔记、提醒的SQLite数据，保留WAL中已提交内容，完成后不可覆盖，中断后可重复原命令续跑。模型、原始音视频和认证文件不进入备份。

[真实备份与故障测试](benchmarks/results/data-backup-smoke.json)验证公开CLI、原生快照和SQLite还原，并完成三个阶段各10次真实SIGKILL续跑。[旧备份基线](benchmarks/results/old-backup-controls-baseline.json)保留了备份后遗忘/取消不能由旧快照体现的失败证据。

恢复入口为`python3 tools/project.py restore-data --backup /备份目录 --current-data-dir /当前数据目录 --data-dir /新目录`。它要求当前控制数据库完整，合并删除、更正、会话和提醒状态，检查通过后禁用旧目录、启用新目录。中断后可续跑，未完成恢复可通过`--abort --data-dir /新目录`撤销。较新的已确认事实丢失时会拒绝回退；待完成写入须先用原请求重试并确认结果。恢复后日常命令须指定新数据目录。详细范围、限制和验证见[数据维护说明](deploy/data/README.md)。

### 人物状态与手势主动回应

启动 `python3 tools/project.py up --components all` 后，`interact` 和 `voice` 默认同时订阅 yolo_ros 的 `/pose/d435/detections_3d`。检测、姿态和跟踪继续由上游 C++/TensorRT 完成；应用只根据跟踪ID和关键点做时序规则。`/scene` 显示匿名人数、方位、有效深度距离及结果年龄，不做人脸识别或根据画面选择主人记忆。

首轮规则为：关键点置信度≥0.6、连续确认≥0.3秒，手腕高于肩部且持续1.2秒形成举手事件；挥手需要1.2秒窗口内两次足够幅度的方向变化。同一手势只形成一次问候，双臂中立姿态持续0.5秒后才重新允许触发。输入消息年龄超过200ms拒绝使用，断流0.5秒后结果失效；最多16个跟踪状态和32个待消费事件。阈值是待真实语料验证的初始设置。

手势问候复用提醒的 SQLAlchemy 持久输出记录及 Pipecat TTS 队列，默认文字为“你好，有什么可以帮你？”。准入冷却2秒，未开始输出的事件2秒后过期；当前对话优先。已开始但未确认的输出记录为 `uncertain`，不会自动重播。应用重启会丢弃旧视觉事件，并保留同一发布者、同一目标的手势锁存，直到看到中立姿态；发布者变化会开启新的跟踪范围。普通提醒查询不显示视觉事件。

`interact` 中用 `/mute-visual on`、`/mute-visual off` 控制手势问候，偏好按当前档案保存；解除手势静音不会补播被关闭的旧事件。两个入口也接受 `--visual-feedback off` 或 `on`，省略参数则沿用偏好。`--no-vision` 同时关闭图像订阅与手势输入。

[端到端报告](benchmarks/results/gesture-feedback-smoke.json)验证了真实D435i/姿态话题连接，以及隔离ROS域内的标注关键点→规则→SQLite输出记录→真实TTS→WAV与ASR回读；覆盖去重、静音、会话占用到过期、输出确认前断流和多发布者拒绝。WAV排空延迟注入明确标记，未产生物理播放回执。真实摄像头初测未观察到人物，**这些证据不证明手势识别准确率，JT-13所需50正例/50负例及实体扬声器验证仍待完成**。早期接口和测试时序问题保留在[接入记录](benchmarks/results/gesture-feedback-integration-baseline.json)。

复现夹具验证（不会向实际运行域发布合成姿态，也不使用日常用户数据库）：

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/AI_Stack/workspaces/vision/install/setup.bash"
.venv/bin/python tools/gesture_feedback_smoke.py
```

### 记忆容量测量

入口 `bench --suite memory-capacity` 已实现，其余B0–B3整机负载和完整 `acceptance` 入口仍待完成。使用[冻结合成语料定义](benchmarks/corpora/memory-capacity-v1.json)，在六个后端驻留、真实检测/姿态持续运行、无并发对话时，对每档100次查询和50次明确事实写入计时。BGE为真实CPU ONNX，查询包括embedding、主体过滤、检索及适配器返回；写入包括SDK、history和operation journal确认。这一原始容量基准使用OpenBLAS1、OMP2、ORT intra2/inter1；当前应用通过`memory.embedding_threads`采用ORT intra4，历史报告按各自实际配置解读。

| 当前Qdrant local基线 | 查询P95 | 完整写入P95 | 容量时延目标 |
| --- | --- | --- | --- |
| [1k档](benchmarks/results/memory-capacity-1k-baseline.json) | 79.2ms | 146.1ms | 该档低于200/500ms参考值 |
| [10k档](benchmarks/results/memory-capacity-10k-baseline.json) | 350.4ms | 906.3ms | **未达到JT-11的200/500ms目标** |

各档另加100条访客硬负例，未观察到主体泄漏，全部50次写入均在计时后检查了正文、历史及journal完成状态。编号近似记录的Top-5命中分别为0/100、86/100；各档查询编号集合不同，召回率仅作各自压力观察。这不是独立200例记忆质量集，1k档低时延也不代表记忆质量或JT验收通过。[本机服务的50k压力档](benchmarks/results/qdrant-native-50k-comparison.json)已完成：查询/完整写入P95为61.6/59.2ms，编号命中94/100，未观察到主体泄漏或未确认写入。该档同样是合成容量观察，正式独立质量和整机验收仍待完成。

隔离的[Qdrant本机服务对照](benchmarks/results/qdrant-native-10k-comparison.json)已完成：相同10k种子记录、ID、向量、BGE与完整写入路径下，查询/写入P95降至47.5/56.6ms，100个查询的目标命中结果逐项一致。使用精确向量扫描及原生字段索引；[40次进程强杀恢复检查](benchmarks/results/qdrant-recovery-smoke.json)通过。文字和语音入口现默认连接受管理的本机Qdrant服务；认证、目录/collection绑定以及实际文字和文件语音已验证，既有嵌入式数据的迁移和完整备份恢复仍待完成。复现方式见[本机服务评估说明](deploy/memory/README.md)。

[分段诊断](benchmarks/results/memory-capacity-10k-profile.json)在已完成数据的隔离副本上进行：10次查询平均总耗时333ms，其中Qdrant查询296ms；10次写入平均846ms，两次Qdrant过滤列表合计约773ms。主耗时在嵌入式存储的过滤/查询路径。上游说明[local模式面向小规模使用，payload索引不生效](https://github.com/qdrant/qdrant-client/blob/v1.19.1/qdrant_client/local/qdrant_local.py)。已评估同一Mem0/Qdrant技术栈的本机服务模式和原生索引，保留此失败基线；公开应用入口已采用本机服务，基准工具仍可显式测量原来的嵌入式路径。

```bash
python3 tools/project.py bench --suite memory-capacity --tier 10000 \
  --seed-method sdk-batch \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/memory-capacity-new-run" \
  --output benchmarks/results/memory-capacity-new-run.json
```

`--run-dir`必须是新目录，或带有完全匹配标记的本工具数据目录；准备阶段中断可在相同目录继续。完成测量会额外写入50条事实，再次完整测量应使用新目录。`adapter`准备模式逐条走完整应用写入；`sdk-batch`仅在隔离测试数据准备阶段复用BGE批量embedding和Mem0的Qdrant/history provider，并写入对应操作日志，减少反复全库冲突扫描。两种准备结果的向量、payload、history、journal指纹已[抽样核对](benchmarks/results/memory-capacity-import-equivalence.json)；计时写入始终使用完整应用适配器。模型、原始数据库和批量测试数据保留在AI_Stack运行目录，不提交Git。

### 依赖与单组件检查

Python依赖使用独立虚拟环境；锁文件记录当前ARM64/Python3.12验证环境。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
python3 tools/project.py doctor
python3 tools/project.py test --suite core
```

`doctor`为只读探测，有未验证外设或缺失依赖时返回非零，并将本机原始报告写入用户数据目录。测试真实调用SDK与数据库；用于embedding/LLM的测试夹具明确标记，不替代模型质量验收。

准备 llama.cpp（首次需要网络、CUDA开发环境和足够磁盘）：

```bash
python3 tools/project.py prepare --component llama --jobs 2
python3 tools/project.py smoke --component llama
```

准备命令固定源码commit、模型revision与SHA256，下载可续传；本机构建与权重分别存放在 `~/AI_Stack/builds/`、`~/AI_Stack/models/`，脚本原缓存路径通过兼容链接访问。smoke临时启动真实llama-server，执行中文文本和合成图像问题，结束后关闭；它只证明加载和推理路径，不等于JT验收。

Pipecat真实文本入口需要本地llama-server已监听 `127.0.0.1:8088`、别名`assistant-vlm`：

```bash
python3 tools/project.py serve --component llama
```

在另一终端运行：

```bash
.venv/bin/python -m app.pipecat_bot --text-smoke \
  --prompt '你好，请用中文回答' \
  --output benchmarks/results/pipecat-text-smoke.json
```

准备并验证本地BGE检索：

```bash
python3 tools/project.py prepare --component memory
python3 tools/project.py smoke --component memory
.venv/bin/python tools/assistant_memory_smoke.py
```

最后一条命令需要llama服务已启动，使用独立临时库及合成偏好，验证保存/更正/遗忘后的真实回答；不读写用户记忆。实际查询指定 `--memory-dir` 与手选 `--user`，记忆证据按本地llama tokenizer限制在512token内。BGE运行使用CPU ONNX，PyTorch CPU仅用于转换和参考回归；锁文件固定官方ARM64 CPU wheel，避免额外安装CUDA运行库。

真实模型工具调用测试使用独立临时库：`.venv/bin/python tools/tool_smoke.py`。当前 llama.cpp 使用其上游 Qwen2.5 工具模板，原 Qwen2.5-VL 内嵌模板未提供工具能力。工具按当前明确请求筛选，参数经校验后交给 Mem0/APScheduler；未收到实际写入成功结果时，Pipecat 输出插件会阻止口头成功确认。三条测试不代表完整工具质量或语音打断验收。

当前Qwen模板在 `tool_choice=required` 时通过 llama.cpp 原生assistant prefill从 `<tool_call>` 开始；普通问答和工具回传后的回答不添加此前缀。该值固定在模型manifest，更换模板时一并核验。预算与推理使用相同的请求转换，前缀不写入会话历史。[固定请求五组对照](benchmarks/results/required-tool-prefill.json)中，原格式5次用完128token仍未调用工具；预填格式5次均返回参数正确的调用。对照只生成工具计划，未执行工具，不替代完整工具质量集或B0–B3性能实验。

工具参数中，明确的记忆原文、笔记标题/正文、提醒事项和时间由应用保留或规范化，再交给实际SDK执行；模型只填写尚未明确的字段。时间复用JioNLP实体/时长解析，按请求时间计算，不让模型自行换算“一个半小时”等数值。候选ID来自真实存储，找不到对应项可以返回不匹配；否定、撤回和模糊目标会阻止写入。写入回执和当前时间回答以工具真实结果为准。

“明天提醒我交报告”会先询问具体时间，随后回答“上午八点”即可补齐；原来的“明天”和“交报告”会保留。待补充请求在当前会话内最多保存5分钟，`/stop`、撤回或切换任务会清除授权，重启后不会自动续办。验证见[补充信息对话报告](benchmarks/results/clarification-smoke.json)和[真实语音文件对话报告](benchmarks/results/voice-clarification-smoke.json)。

[100例工具开发语料](benchmarks/corpora/README.md)使用实际模型、BGE和临时SDK数据库，逐例检查意图、参数与状态。[原始基线](benchmarks/results/tool-quality-development-baseline.json)为63/100，[首轮修复](benchmarks/results/tool-quality-development-iteration1.json)为99/100；[最新结果](benchmarks/results/tool-quality-development.json)为100/100，保留全部逐项记录。这是开发集，未替代独立验收和真实提醒播放测量。

音频设备确定为ReSpeaker Lite双麦，用户当前无法连接麦克风/扬声器，真机录播/AEC/打断列为待验证。D435i最新链路为5Gbps；早期USB2枚举仅保留为排查记录。

视觉数据接入已完成真实D435i→ROS→有界RGB-D缓存→Pipecat图像请求帧的初测。缓存使用2秒窗口，128MiB总预算中为传输与转换预留16MiB；仍被消费者引用的帧继续计入缓存占用。按提问时刻选择150ms以内的画面，停止采集后拒绝旧画面，驱动重启会更换数据代次。21次图像请求及一次驱动重启检查见[视觉接入报告](benchmarks/results/vision-bridge-smoke.json)。本项未包含检测推理或VLM图像问答，不代表JT-03已通过。

从项目根目录验证上述接入（脚本会启动、重启并关闭其拥有的相机进程，不保存原始图像）：

```bash
source /opt/ros/jazzy/setup.bash
.venv/bin/python tools/vision_bridge_probe.py --launch-camera
```

语音与视觉准备细节分别见 [语音部署说明](deploy/speech/README.md)、[视觉部署说明](deploy/vision/README.md)。`interact` 已实现文本入口；`up/down/status/logs`通过systemd管理所选后端，使用方法见[后端生命周期说明](deploy/systemd/README.md)。`backend_status=READY`只表示后端接口就绪，完整助手与物理设备仍待验收。`bench`目前提供记忆容量工作负载，完整B0–B3与`acceptance`入口仍待实现。

## 数据和上游

本机运行数据位于 `~/AI_Stack/runtime-data/`，原路径 `~/.local/share/multi-model-assistant/` 是兼容入口，通过 `ASSISTANT_DATA_DIR` 覆盖；缓存根目录通过 `ASSISTANT_CACHE_DIR` 覆盖。仓库提交源码、配置、模型清单和脱敏验证报告。现有 `references/` 保留。

上游版本见 [依赖锁](deploy/upstreams.lock.json)。Mem0的局部历史及实体关联清除补丁位于 `patches/mem0/`，用于落实显式遗忘；重启后仍会检查实体索引，清理失败保留删除屏障以便重试。SDK的Qdrant/SQLite操作不是跨存储原子事务，适配器通过持久操作状态与删除屏障恢复，仍需整机故障验收。原始自建运行时/记忆原型已归档到本机缓存，不参与当前产品。

### 独立来源的中文记忆质量评测

已接入官方CLongEval的LCvMem数据和评分方法，按对话冻结82题开发集、200题保留集、76题备用集。20个开发问题的真实BGE/Mem0/Qdrant/llama/Pipecat评测中，去除提示内完全重复的来源摘录后，词级F1由12.92升至16.18；加入原始发言日期、说话者和检索条件后为24.73，含参考原文的注入样例由6题增至10题，无意外写入或跨主体证据。后一次同时改变多个因素，词级指标也不是语义正确率；200题保留集未运行，JT-10未通过。方法、复现命令及原始对照见[评测说明](deploy/evaluation/README.md)。

普通对话中的显式保存、更正会记录本轮的本地日期和用户角色。明确发言日的历史问题可在检索前筛选日期及说话者；未写年份时保留月日条件和证据中的实际年份。公开文本入口的[合成验证](benchmarks/results/memory-sources-console-smoke.json)覆盖重启后回忆、角色区分、缺失日期及主体隔离。相对日期、范围、事件发生时间和事实历史版本仍待实现；没有日期的旧记忆不会自动补标签。

最新[20题开发对照](benchmarks/results/lcvmem-reference-comparison.json)在保持top5和512-token预算时，将完整标识符换成本轮短引用，并保留实际ID、版本和来源映射；更正/遗忘继续使用完整ID。平均注入证据由2.65增至4.7条，含参考原文的注入题数由10增至14；“我跟你分享”的说话者误判也已修复。新基线F1为21.17，最终为27.05，均无意外写入。旧格式重跑仍有回答变化，词级分数不能当成稳定正确率。198项框架回归和公开保存/更正/遗忘闭环通过。[82题完整开发集](benchmarks/results/lcvmem-development-all-refs.json)已完成，词级F1为26.33，错误/意外写入/跨主体证据均为0；200题保留集尚未使用。

[长条目失败基线](benchmarks/results/long-memory-feasibility-summary.json)中，8份约3.1KiB资料的尾部被嵌入截断，语义top5命中5/8，完整证据均未装入512-token预算。现已将原生多向量和BGE ARM64 int8接入正式适配器，新建档案默认启用，已有档案保留原绑定并用`reindex-memory --data-dir ...`显式升级。规范ID、完整正文、版本和来源保持一致，提示使用可核验的原文片段。

[公开入口集成](benchmarks/results/window-legacy-backup-integration.json)验证了长资料保存、跨会话回忆、迁移、更正和旧格式备份恢复，已遗忘内容未复活；[30次迁移SIGKILL](benchmarks/results/window-reindex-recovery.json)均可续跑，完成后重试不会回放已删除记录。单次约3.1KiB资料的完整适配器写入观测约0.36秒，不代表10k规模或整机时延验收。准备、升级及撤销方法见[记忆部署说明](deploy/memory/README.md#长条目索引与已有档案升级)。

新格式的[82题开发评测](benchmarks/results/lcvmem-development-windows.json)已完成：词级F1为26.69，归一化完全匹配1/82，参考原文实际注入56题，执行错误、意外写入及跨主体证据均为0。旧格式F1为26.33，这个小幅差异不足以证明整体质量提升，200题保留集未使用。旧格式备份恢复到当前格式的[40次SIGKILL](benchmarks/results/window-restore-fault-recovery.json)全部续跑成功，保留当前更正和新增事实，未复活已遗忘内容；这轮没有生成模型回答。

[万条混合长短记忆计时](benchmarks/results/managed-window-capacity-10k.json)查询P95约47ms、完整写入约343ms，无执行错误或主体泄漏，50次写入均持久确认。但编号top5召回只有9/100；[诊断](benchmarks/results/numbered-window-recall-diagnostic.json)确认原文完整，长目标却没有进入前60候选。随后在独立索引复用Qdrant本地BM25及原生RRF/DBSF，[三种候选均命中100/100](benchmarks/results/qdrant-lexical-numbered-feasibility.json)；这些属于合成可行性结果。现已将原生DBSF接入可显式选择的`bge-int8-windows-bm25-v1`候选，保留Mem0存储与journal；[实际模型集成](benchmarks/results/hybrid-memory-integration.json)通过长资料保存/召回、更正、旧备份恢复和遗忘，217项框架测试通过。默认仍用原窗口配置。候选已通过[30次迁移SIGKILL](benchmarks/results/hybrid-reindex-recovery.json)；[万条正式适配器测试](benchmarks/results/managed-hybrid-capacity-10k.json)编号命中100/100，查询/完整写入P95约62/356ms，无错误或主体泄漏。82题候选开发评测和证据预算对照已完成，见下文；这些仍不是独立正确率或完整语音共存验收。

同一万条个人档案下的[语音/相机/记忆并发基线](benchmarks/results/memory-voice-load-10k.json)完成20轮语音、106次查询与53次写入：查询/写入P95约127/459ms，检测/姿态约11.64Hz，最新年龄P95约174/178ms，以上检查通过。偏好/时间问答首段PCM P95约3.09/3.10秒，图像问答约5.55秒，超过3/5秒目标，整轮报告为FAIL。输出为WAV；每类问题仅5轮，没有物理声学或画面准确率结论。独立质量、完整负载和JT-10/11继续待验收。

目前完整开发对照已进一步定位到检索配置和提示装配的固定上限。配置现在贯通文字、`/new`和语音入口，候选数有20条硬上限，实际注入继续受token预算约束；默认仍为原窗口配置、5条/512tokens。DBSF的[8条/768tokens开发结果](benchmarks/results/lcvmem-development-hybrid-top8-768.json)F1为29.58，参考原文实际注入65/82题，没有执行错误、意外写入或跨主体证据；不等于语义正确率达标。此前[5条/512tokens候选语音共存](benchmarks/results/memory-voice-hybrid-10k.json)短问答首段文件音频P95最高2.902秒，但图像为5.202秒，整轮仍FAIL。完整对照、无效实验与范围见[评测说明](deploy/evaluation/README.md#候选开发质量与证据预算)。

后续资源检查发现llama.cpp默认主机提示缓存上限每服务8GiB，日志中VLM已积累约3.25GiB，ASR约431MiB。现显式限制为256/64MiB，保持2K上下文；真实请求已验证缓存淘汰及语音保存/回忆，222项回归通过。[8条/768tokens并发结果](benchmarks/results/memory-voice-hybrid-8x768.json)图像首段文件音频P95为4.895秒，但短问答最高3.271秒，整体仍FAIL。[阶段追踪](benchmarks/results/voice-stage-diagnostic.json)显示慢算术请求携带707tokens记忆证据、模型首token等待约1.67秒；后续须验证选择性检索，当前没有宣称已解决短问答时延或最大事件循环延误。候选[40次跨格式恢复故障](benchmarks/results/hybrid-restore-fault-recovery.json)全部通过，报告保留了测试中模型服务的内存干预记录。

完整字面算式可通过默认关闭的实验开关省略长期检索，保留当前会话、来源依赖并继续调用模型；个人/指代/单位等问题不走捷径，225项回归通过。固定到达对照中，一轮在图像输入epoch变化时正确拒绝旧帧并失败；全新档案的GC诊断复跑完成20轮，算术P95约1.94秒，但回忆约4.35秒，整体仍未达标。[诊断](benchmarks/results/voice-gc-pause-diagnostic.json)发现一次约580ms的第二代GC与573ms事件循环延误重叠。GC策略未改，选择性检索也未设为默认；后续验证静态对象管理及长时稳定性。

GC启动处理的[独立文件进程对照](benchmarks/results/voice-heap-comparison.json)已完成：只回收与回收后冻结都完成20轮，自动GC和阈值保持不变，关闭后恢复永久代基线。两轮运行期间均未出现第二代GC，回忆首段音频P95仍约4.3秒，尚未证明冻结带来整体收益，也未启用为默认。227项回归通过；长时内存与冷启动延迟继续验证。
