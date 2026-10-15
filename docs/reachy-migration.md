# Reachy隔离迁移：当前操作入口与边界

2026-10-15。当前只交付R1的文字/文件图像核心，不代表R1或完整应用完成。

## 目录与来源

- 运行旧系统：`/home/nvidia/Multi_Model_Inference`，仍在`main`。
- 新派生工程：`/home/nvidia/AI_Stack/projects/reachy-headless`，分支`migration/reachy-headless`。
- 新Python环境：`/home/nvidia/AI_Stack/environments/reachy-headless`，不继承旧环境的site-packages。
- 只读上游：`/home/nvidia/AI_Stack/research/reachy-mini-jetson-assistant-2673202`。
- 旧系统快照：tag `legacy-before-reachy-v3-20261015`；可验证bundle和配置在`/home/nvidia/AI_Stack/archives/reachy-r0-20261015`。
- [导入清单](../deploy/reachy-import.json)记录46个原样导入文件的SHA256。提交`2084ded`保留裁剪前原样导入；随后差异可直接比较。旧源码、工具、测试保留于旧标签，未删除运行目录或数据。

## 已可运行的入口

```bash
cd /home/nvidia/AI_Stack/projects/reachy-headless
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py config
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py ask '只回答：连接成功' --json-output
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py ask '图中的主要交通工具是什么？' --image /home/nvidia/AI_Stack/runtime-data/benchmarks/vision-input-diagnostic-v1/decoded.png --json-output
```

这是单次请求接口，尚不提供多轮会话、工具或物理语音。图像例子使用历史固定测试图片，不是实时摄像头。JSON中的单次耗时用于诊断，不是冻结的性能基线。

当前`run_web_vision_chat.py`、`run_vision_chat.py`和`run_voice_chat.py`仍在迁移，尚不是可用启动入口；它们在适配完成前会明确退出。后续统一在上游网页/音视频主线继续适配，不建立第二套主控。

## 配置与模型

沿用上游typed dataclass和overlay组织。默认加载`config/settings.yaml`及`config/settings.orin-nx-headless.yaml`；显式`--config`或`REACHY_ASSISTANT_CONFIG`可选择另一个完整平台overlay。未知字段、错误类型、缺失文件、机器人/RAG启用和远端模型地址均明确失败。输出的effective config是当前参数依据。

上游YAML中的`vad.use_silero`未被原dataclass消费，已删除这个无效配置项；真正的VAD接入与验证在后续阶段进行。

模型使用已有`assistant-vlm`本机服务。主模型由既有模型锁固定为Qwen3.5-4B Q4_K_M。客户端每次生成前使用同一请求参数调用原生多模态token计数端点，保留128输出和16 token余量；不按字符估算。显式模型名称不匹配时不自动替换；HTTP失败、异常结束和空输出不能伪装成功。

## 资源所有权

| 资源 | 当前所有者 | 新应用策略 |
|---|---|---|
| llama.cpp 8088 | 旧systemd `mmassistant-llama` | 只连接，不重复启动或终止 |
| ASR 8089 / 语音8621 | 旧systemd `mmassistant-asr` / `mmassistant-speech` | R2再通过接口适配，当前CLI不调用 |
| D435i | 旧systemd `mmassistant-camera` | 尚未切换；后续受控停止自己的旧采集并在测试后恢复 |
| 原视觉/姿态/Qdrant | 旧项目服务 | 当前保持原状；新应用不导入ROS/Pipecat/记忆 |
| 麦克风/扬声器 | 当前未接入指定实物 | 不生成采集或播放成功回执 |
| 新Python客户端 | 每次CLI请求 | 关闭自身HTTP流，释放后检查模型槽空闲 |
| Web端口8090 | 尚未启动 | 后续只管理本项目实例 |

所有新安装仅进入独立环境。不更换BSP、CUDA或TRT，不运行上游清理其他相机/音频进程的函数。依赖锁为`requirements.lock`，当前仅覆盖R1核心及开发测试；不包含尚未验证的GPU Python或语音模型依赖。

## 验证与未完成项

[核心环境与回归](../benchmarks/results/reachy-r1-core-environment.json)、[原样上游客户端参考](../benchmarks/results/reachy-r1-reference-text.json)、[真实文字/文件图像/容量拒绝](../benchmarks/results/reachy-r1-cli.json)。21项针对性测试通过；原上游其余测试和完整应用尚未适配，未宣称全套通过。

尚待完成：网页与采集入口、人脸和运动逻辑解耦、机器人/RAG源码和配置裁剪、中文语音adapter、人体/手部感知、统一轮次取消及资源释放、实机验收和B0基线。

回退时退出新应用即可继续旧系统。默认服务未切换；如将来测试暂停旧相机，必须记录并恢复原服务状态。不要在运行目录执行`git reset --hard`来替代受控回退，不覆盖原用户数据。本次快照保存代码/配置，不宣称已备份在线数据库。
