# 当前 B0 固定语料

- `b0-speech-v1.json`：20 条中文语音合成提示及归一化口径。生成 WAV 与输入哈希位于 `05_runtime/datasets/b0-sw-v1/`。提示文本与转写的一致性不等于独立真人 ASR 准确率。
- `b0-llm-v1.json`：文字与图文固定请求，供单模型和完整应用基准使用。

采集入口为 `01_src/benchmarks/model_modules.py` 和 `01_src/benchmarks/application_workload.py`，测量口径见 [冻结协议](../manifests/b0-sw-v1.json)；这些输入不由业务代码读取。

旧长期记忆、工具质量、推测解码语料已归档，原始内容可从 [清理恢复索引](../../docs/archive/repository-cleanup-20261019.json) 对应的 Git 标签恢复。
