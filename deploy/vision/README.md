# 首版视觉服务的准备与构建

固定 **YOLOv8n 检测 + YOLOv8n-pose 姿态**，接入 yolo_ros 5.0.0 的 C++/ONNX Runtime 管线，使用 TensorRT execution provider。姿态按计划在 J3 接入。当前完成两模型CPU导出回归及检测TensorRT执行；ROS模型整合仍待完成，ORT GPU依赖正在编译。

## 已核验的版本与限制

| 项目 | 固定值或状态 |
|---|---|
| yolo_ros | `7924a8be76c81f4051370e254677516ca696a272`，main / Jazzy |
| ONNX Runtime | `v1.28.0`，tag commit `da9b5e364c465de65c49d91e696cd6485270757f` |
| GPU 目标 | Orin SM87、CUDA13.2、当前 TensorRT10.16 |
| ORT 构建 | CUDA + TensorRT、shared library、先用 2 个编译任务 |
| ROS | Jazzy与realsense2-camera已安装；双路RGB-D采集实测通过 |
| YOLO ONNX 模型 | 两模型已固定权重哈希并导出640×640、opset17、无NMS的FP32图；CPU回归通过 |
| 检测TensorRT引擎 | FP16单图框位置对齐失败已保留；FP32关闭/允许TF32两种配置均通过原0.98 IoU门槛，后续ROS采用FP32 |
| 姿态TensorRT引擎 | FP32允许TF32实测通过：3人、可见关键点最大坐标差0.053像素以内；不是完整姿态质量验收 |
| ROS构建 | 五个包已用同版本ORT核心SDK编译通过；完整GPU provider库仍在构建，尚未启动GPU ROS节点 |

上游 vendor 的 GPU 预编译 tarball 只支持 x64。aarch64 GPU 构建必须使用源码 ORT，并将 `-DONNXRUNTIME_ROOT` 指向编译输出。CUDA13 对应的上游选择为 ORT1.28.0。[固定版构建说明](https://github.com/mgonzs13/yolo_ros/blob/7924a8be76c81f4051370e254677516ca696a272/docs/build.md)

`provider=tensorrt` 是显式选择，仍然存在 TensorRT → CUDA → CPU 回退。启动日志中仅看到接受 session 的主 provider 不能证明整个模型每个节点都在 TensorRT/GPU 上；需保存 ORT profiling 和实际运行证据。GPU 基线不能由 CPU 功能测试替代。

## 可执行的准备与构建命令

从项目根目录运行。源码与产物已统一放在 `~/AI_Stack/`；旧缓存路径通过兼容链接访问，见[依赖总目录说明](../dependencies-layout.md)。

```bash
python3 deploy/vision/prepare.py --fetch --stage-sdk \
  --output "$HOME/.cache/multi-model-assistant/evidence/vision-source-inspection.json"
```

此命令固定 yolo_ros revision，并创建 CUDNN/TensorRT SDK 的用户缓存 symlink prefix，适配本机 `/usr/include/aarch64-linux-gnu` 与 `/usr/lib/aarch64-linux-gnu`。它不安装系统包、不编译、不启动模型。

确认其它大模型构建/加载已结束后，开始 ORT 构建：

```bash
python3 deploy/vision/prepare.py --build-ort
```

脚本下载 ORT v1.28.0 及 submodules，校验 tag commit，使用 CUDA13.2、SM87、TensorRT/CUDNN SDK prefix 和 2 个并发编译任务。可选attention实现设为关闭，但该版CMake仍会编译其共用CUDA对象，不能据此宣称构建已裁剪。修复补丁和SHA256纳入[source-lock](source-lock.json)：可选注意力关闭时仍暴露原有mask常量、打包ORT1.28拆分的C API头文件，以及按需启用原生ORT profiler。补丁未改变mask常量的数值。

构建目录迁移后，上游Python构建入口解析到新的规范路径，重新配置会使部分已有对象失效；保留目录不等同于保证所有对象可复用。已记录一次并行“ORT编译+ROS编译+驻留VLM”导致的[OOM失败](../../benchmarks/results/build-memory-failure.json)。当前`serve`在重型ORT构建存活时拒绝启动语言/语音模型服务；先完成重型构建，再测运行时容量和并发。该安装阶段故障不会被算成运行时验收通过。

安装 ROS2 Jazzy Noble ARM64 发行包、colcon、rosdep，并满足 yolo_ros 各 package.xml 的依赖后：

```bash
python3 deploy/vision/prepare.py --build-ros
```

该命令只创建缓存中的 vision workspace 并串行构建；依赖不齐时会明确失败。系统 CUDA/TensorRT 包不得被上游通用 x64 安装脚本替换。

ORT尚未完成时可以先构建消息接口与固定版本的Hugging Face C++依赖：`python3 deploy/vision/prepare.py --build-ros-prereqs`。当ORT核心库已生成，`--stage-ort-core --build-ros`可用独立的编译SDK提前编译ROS代码。这个SDK不代表GPU运行时已就绪；最终仍须执行不带`--stage-ort-core`的`--build-ros`，服务启动会检查完整CUDA/TensorRT provider产物和上游构建标记。

## 本机服务入口

```bash
python3 tools/project.py serve --component camera
python3 tools/project.py serve --component vision
python3 tools/project.py serve --component pose
```

三项分别在独立终端前台运行。相机启用640×480、30Hz彩色与对齐深度。视觉节点复用上游Detection、ByteTrack、Detect3D插件，模型输入640×640、batch1、max_fps15，固定本地权重并禁用下载。ROS使用独立domain42及LOCALHOST发现范围；`ASSISTANT_ROS_DOMAIN_ID`可覆盖domain。彩色与深度的最新[对齐采集报告](../../benchmarks/results/camera-aligned-smoke.json)通过格式、坐标系、内参和时间戳配对检查，物理深度精度没有地面真值验证。

检测/姿态参数生成在运行数据目录的`config/ros-*.json`。当前GPU ORT构建未完成时，后两项会明确拒绝启动。需要原始provider trace时直接使用`python3 deploy/vision/serve.py detection --profile`或`pose --profile`，并正常结束进程以刷新trace；默认不启用逐节点profiling。

## 模型、相机与调度接入

模型导出使用独立环境：

```bash
python3 -m venv .venv-vision
.venv-vision/bin/python -m pip install -r deploy/vision/requirements.lock
.venv-vision/bin/python deploy/vision/export_models.py
.venv-vision/bin/python tools/vision_engine_smoke.py
.venv-vision/bin/python tools/vision_engine_smoke.py --precision fp32_tf32
.venv-vision/bin/python tools/vision_engine_smoke.py --task pose --precision fp32_tf32
```

最后一条命令调用本机NVIDIA `trtexec`构建并实际执行TensorRT引擎，再用Ultralytics NMS比较CPU ONNX和GPU输出。已有引擎可加`--reuse-engine`重新执行；每次记录实际模型/引擎/输入哈希。公开`bus.jpg`初测均检出3人和1辆公交车，人物框IoU≥0.998，公交车框IoU为0.973，未达到预先设置的0.98，因此[对齐报告](../../benchmarks/results/yolo-engine-smoke.json)保留FAIL。不能将“引擎执行成功”作为全部数值/质量验收通过；单次耗时也不能当成吞吐或P95验收。

排查发现FP16把公交车的两个候选分数舍入为相等值，NMS选择了不同框。门槛不变时，[严格FP32](../../benchmarks/results/yolo-engine-fp32-smoke.json)和[允许TF32的FP32](../../benchmarks/results/yolo-engine-fp32_tf32-smoke.json)均通过，后者最小IoU为0.999789；[姿态对齐](../../benchmarks/results/yolo-pose-engine-fp32_tf32-smoke.json)也通过。TF32允许使用与实际采用哪种tactic是不同的观测项；NVIDIA说明TensorRT默认允许TF32，诊断时可用`--noTF32`关闭。[精度控制说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)

1. 固定 Ultralytics exporter 版本与 YOLOv8n/YOLOv8n-pose 权重 revision/hash；在独立环境导出 static shape、batch=1、明确 raw-output/NMS 规则的 ONNX。
2. 将权重许可、ONNX SHA256、输入形状、类别/关键点定义和 exporter 参数纳入模型锁。yolo_ros 的 MIT 许可不代替 YOLO 权重许可。
3. 先使用真实图片/文件输入验证检测、姿态、后处理和执行 provider；保存质量样例与 GPU trace。
4. 接入 realsense-ros D435i 的 RGB/depth/CameraInfo。2026-10-12 后续 probe 已发现 D435i `8086:0b3a` 和 `/dev/video0`～`video5`，枚举速度为 **480 Mbit/s（USB2）**；这是早期枚举记录；后续链路已变为5Gbps。已用ROS实测640×480双路20s，RGB29.48Hz、depth29.78Hz，见[采集报告](../../benchmarks/results/camera-smoke.json)。跨流同步质量、视觉模型及长时稳定性仍需测试。
5. 上游已移除旧 `/yolo/enable` 服务，推理持续运行。图像准入应接入 `camera_streams.cpp` / detection plugin 输入，执行回调应围绕 `engine/model.cpp` 的实际 `Ort::Session::Run` 与同步完成边界补充。
6. detection plugin 已有 latest-frame-wins 路径。接入 Pipecat 视觉桥接插件后同时记录丢帧原因、帧时间戳、开始/结束、provider 和 worker epoch，避免把“输入已丢弃”当作“GPU 已释放”。

J0.3 通过要求：真实模型执行、实际 TensorRT/GPU 后端证据、真实相机图像、峰值内存与帧时效记录。当前只完成上述部分证据，未通过完整J0.3验收。
