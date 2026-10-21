# Orin NX 端侧多模型服务

当前目录只保留运行版本：D435i 原生 RGBD、十个 TensorRT 视觉模型、4B 图文模型、ASR/TTS、Web 对话和板端实时状态。历史版本、备份、学习资料、构建工具、测试与性能分析工具已删除。

## 启动

```bash
cd /home/nvidia/Multi_Model_Inference
./run.sh services check
./run.sh services up
./run.sh services status
```

四个服务均为 `READY` 后，打开 `http://127.0.0.1:8090/`；`/debug` 显示相机、模型调用和板端实时指标。远程访问方式、停止与日志命令见 [运行操作](docs/operations.md)。

默认配置为 `02_config/settings.orin-nx.yaml`，使用 `./run.sh config` 查看有效参数。当前音频采用文件模式，实体麦克风/扬声器尚未验收。

## 目录

```text
01_src/app/       应用、采集、视觉、对话、语音、动作存储和 Web
01_src/deploy/    启动命令、服务管理、运行文件校验和语音适配
02_config/       当前配置
03_models/       现用模型及引擎契约
04_web/          对话与实时状态页面
05_runtime/      现用二进制、环境、模型、框架运行文件与业务数据
docs/            运行操作、软件架构、模型说明
licenses/        必须保留的第三方许可证
```

`05_runtime` 不进入 Git，当前板子已具备运行文件。Git 克隆本身不包含模型和环境；本项目不再保存运行包、恢复副本或构建流程。

[运行操作](docs/operations.md) · [软件架构](docs/project-overview.md) · [现用模型](docs/models.md)
