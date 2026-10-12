# 首版视觉服务的准备与构建

固定 **YOLOv8n 检测 + YOLOv8n-pose 姿态**，接入 yolo_ros 5.0.0 的 C++/ONNX Runtime 管线，使用 TensorRT execution provider。姿态按计划在 J3 接入。当前只完成源码与构建依赖核验，模型推理尚未验收。

## 已核验的版本与限制

| 项目 | 固定值或状态 |
|---|---|
| yolo_ros | `7924a8be76c81f4051370e254677516ca696a272`，main / Jazzy |
| ONNX Runtime | `v1.28.0`，tag commit `da9b5e364c465de65c49d91e696cd6485270757f` |
| GPU 目标 | Orin SM87、CUDA13.2、当前 TensorRT10.16 |
| ORT 构建 | CUDA + TensorRT、shared library、先用 2 个编译任务 |
| ROS | 本机 `/opt/ros/jazzy/setup.bash` 尚不存在，需安装及构建依赖 |
| YOLO ONNX 模型 | 尚未下载/导出、尚未锁定哈希；不能作为已部署模型 |

上游 vendor 的 GPU 预编译 tarball 只支持 x64。aarch64 GPU 构建必须使用源码 ORT，并将 `-DONNXRUNTIME_ROOT` 指向编译输出。CUDA13 对应的上游选择为 ORT1.28.0。[固定版构建说明](https://github.com/mgonzs13/yolo_ros/blob/7924a8be76c81f4051370e254677516ca696a272/docs/build.md)

`provider=tensorrt` 是显式选择，仍然存在 TensorRT → CUDA → CPU 回退。启动日志中仅看到接受 session 的主 provider 不能证明整个模型每个节点都在 TensorRT/GPU 上；需保存 ORT profiling 和实际运行证据。GPU 基线不能由 CPU 功能测试替代。

## 可执行的准备与构建命令

从项目根目录运行，源码和产物保存在用户缓存。

```bash
python3 deploy/vision/prepare.py --fetch --stage-sdk \
  --output "$HOME/.cache/multi-model-assistant/evidence/vision-source-inspection.json"
```

此命令固定 yolo_ros revision，并创建 CUDNN/TensorRT SDK 的用户缓存 symlink prefix，适配本机 `/usr/include/aarch64-linux-gnu` 与 `/usr/lib/aarch64-linux-gnu`。它不安装系统包、不编译、不启动模型。

确认其它大模型构建/加载已结束后，开始 ORT 构建：

```bash
python3 deploy/vision/prepare.py --build-ort
```

脚本下载 ORT v1.28.0 及 submodules，校验 tag commit，使用 CUDA13.2、SM87、TensorRT/CUDNN SDK prefix 和 2 个并发编译任务。attention kernels 默认关闭，适用于当前 YOLOv8 检测/姿态模型。保留 build 目录供中断后继续；失败结果不能算作已构建。

安装 ROS2 Jazzy Noble ARM64 发行包、colcon、rosdep，并满足 yolo_ros 各 package.xml 的依赖后：

```bash
python3 deploy/vision/prepare.py --build-ros
```

该命令只创建缓存中的 vision workspace 并串行构建；依赖不齐时会明确失败。系统 CUDA/TensorRT 包不得被上游通用 x64 安装脚本替换。

## 模型、相机与调度接入

1. 固定 Ultralytics exporter 版本与 YOLOv8n/YOLOv8n-pose 权重 revision/hash；在独立环境导出 static shape、batch=1、明确 raw-output/NMS 规则的 ONNX。
2. 将权重许可、ONNX SHA256、输入形状、类别/关键点定义和 exporter 参数纳入模型锁。yolo_ros 的 MIT 许可不代替 YOLO 权重许可。
3. 先使用真实图片/文件输入验证检测、姿态、后处理和执行 provider；保存质量样例与 GPU trace。
4. 接入 realsense-ros D435i 的 RGB/depth/CameraInfo。2026-10-12 后续 probe 已发现 D435i `8086:0b3a` 和 `/dev/video0`～`video5`，枚举速度为 **480 Mbit/s（USB2）**；相机实际帧率、同步和 RGB/depth 共存能力仍需测试。建议实物连接至 USB3 数据链路后重新核验。
5. 上游已移除旧 `/yolo/enable` 服务，推理持续运行。图像准入应接入 `camera_streams.cpp` / detection plugin 输入，执行回调应围绕 `engine/model.cpp` 的实际 `Ort::Session::Run` 与同步完成边界补充。
6. detection plugin 已有 latest-frame-wins 路径。接入 Pipecat 视觉桥接插件后同时记录丢帧原因、帧时间戳、开始/结束、provider 和 worker epoch，避免把“输入已丢弃”当作“GPU 已释放”。

J0.3 通过要求：真实模型执行、实际 TensorRT/GPU 后端证据、真实相机图像、峰值内存与帧时效记录。当前准备脚本没有生成此类推理通过结果。
