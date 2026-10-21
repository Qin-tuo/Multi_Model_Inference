# 运行依赖、构建依赖与独立恢复

当前主线在项目自己的 `05_runtime/` 中运行。系统仍提供兼容的 Jetson Linux、CUDA、TensorRT、Python 与设备库；运行包是当前主机工件的可校验交付，不是操作系统镜像。

| 路径 | 用途 |
|---|---|
| `environments/edge-assistant/`、`environments/speech/` | 应用与语音环境 |
| `builds/llama/bin/` | 已锁定的原生推理程序和共享库 |
| `frameworks/llama.cpp/`、`frameworks/openvoicestream/` | 固定上游源码、模板与语音服务 |
| `models/`、`frameworks/b1/`、`frameworks/b2/` | 被当前清单选择的模型、引擎和辅助代码；物理目录名保留来源信息 |
| `datasets/`、`baselines/` | 仍需使用的固定图片、WAV、参考样本与冻结配置 |
| `runtime-data/` | 服务配置、业务数据库、图库、输出音频与日志 |
| `validation/`、`refactor-checkpoints/` | 验证、诊断和重构恢复证据 |
| `exports/` | 运行包暂存和归档 |

运行包选择器读取当前模型清单，保存引擎、ONNX、运行辅助代码、环境、原生程序、许可证及复现输入；不把全部旧模型和实验目录打包。开发用 PyTorch/OpenVINO、导出环境、原始 checkpoint 和完整参考资产继续保留在本机，但不属于生产推理依赖包。

## 当前状态

当前包已完成同机独立恢复：隐藏原项目和旧 AI_Stack、仅启用 loopback，文字、看图、文件语音、D435i 和十模型加载通过；四个进程均以退出码 0 结束。包路径、SHA256、代码提交与证据见 [运行包锁](project-runtime.lock.json)。它不替代实体声学、真人准确率或长时间稳定性验收。

## 制作与恢复流程

先提交当前源码/清单并冻结输入。打包命令会校验协议、模型和输入摘要，不覆盖已有暂存目录或归档：

```bash
source ./05_runtime/environments/edge-assistant/bin/activate
python scripts/runtime.py package   --stage 05_runtime/exports/current-stage   --output 05_runtime/exports/current-runtime.tar
```

新副本使用匹配代码提交，目标 `05_runtime` 必须不存在。恢复入口仅依赖 Python 标准库，因此可先用系统 Python 运行：

```bash
python3 scripts/runtime.py bundle unpack   --archive /path/to/current-runtime.tar   --sha256 <锁中记录的SHA256>
./run.sh services check
```

恢复器检查归档和逐文件摘要，重建已声明的主机链接，再调整虚拟环境入口路径。框架、模型和业务参数不通过隐式下载或降级来补齐。

独立验证使用 `scripts/validate.py restore --checkout <副本> --output <验证目录>`。副本和输出必须在原项目及旧 AI_Stack 路径之外；验证在私有挂载/网络命名空间隐藏原路径，仅启用 loopback，执行相机、十模型加载、文字/文件语音和进程停止，再恢复原项目服务。副本不注册系统服务。它证明同一兼容主机上的独立恢复，不证明跨 BSP 或跨硬件直接兼容。

新包通过独立验证后，才清理被替代的运行归档和有依据的无引用工件。用户上传的原始 ONNX、参考资料、许可证、整合前 B2 输入/基准及必要失败证据继续保留。
