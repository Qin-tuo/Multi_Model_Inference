# 视觉基线复现入口

先从项目根目录启动完整程序，确认相机与四个服务正常，再理解各模块职责。当前名称为“多模型端侧助手”，服务前缀为 `edge-assistant`。

```bash
./run.sh services up
./run.sh services status
```

下面两个脚本分别测量 V1 单视觉模块和 V2 连续视觉链路。它们会临时停止或切换本项目服务，结束后恢复原配置；每次使用新的输出目录，测量期间不要同时运行其他测试、构建或下载任务。

```bash
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_modules.py \
  --output 05_runtime/validation/vision-v1-new
./05_runtime/environments/edge-assistant/bin/python 01_src/benchmarks/vision_stream.py \
  --output 05_runtime/validation/vision-v2-new
```

固定输入在 `05_runtime/datasets/b0-sw-v1/`。脚本校验 [B0 协议](../benchmarks/manifests/b0-sw-v1.json)、原始配置、模型及输入哈希，路径兼容层将历史目录映射到当前目录；冻结记录保持原字节。

对照 [原始视觉汇总](../benchmarks/results/b0-vision-summary.json) 阅读结果，关注分模块 P50/P95、感知完成率、结果帧龄、无结果比例与资源占用。当前相机配置 10 FPS、感知配置 5 FPS；不要将配置值当作最大吞吐，也不要把网页刷新速度当作板端推理速度。后续由你亲自复现、提出假设，再逐项分析和优化。
