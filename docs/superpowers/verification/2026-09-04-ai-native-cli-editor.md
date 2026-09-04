# AI-Native CLI Editor 验收记录

日期：2026-09-04  
平台：Ubuntu 24.04 x86_64

## 版本清单

| 工具 | 验证结果 |
| --- | --- |
| Neovim | `NVIM v0.12.5` |
| Codex CLI | `codex-cli 0.153.2` |
| codex-acp | `@agentclientprotocol/codex-acp 1.8.0` |
| clangd | Ubuntu clangd 18.1.3 |
| clang-format | 用户级 LLVM 18.1.3 wrapper |
| Python | 3.12.x |
| Ruff | 0.16.5 |
| Pyright | 1.1.413 |
| debugpy | 1.8.21 用户虚拟环境 |
| CMake / GDB | 3.28.3 / 15.1 |
| ripgrep / fd / jq | 14.1.0 / 10.5.0 / 1.7.1 |
| lazygit | 0.64.1 |

## 自动化验证

在 Mars.nvim 实现 worktree 中执行：

```text
bash tests/context.sh                         -> pass
bash tests/ai_helper.sh                       -> AI helper checks passed
bash tests/git_workbench.sh                   -> Git workbench checks passed
bash tests/resume.sh                          -> Codex resume checks passed
bash tests/debug_workbench.sh                 -> Debug workbench checks passed
bash tests/lint_workbench.sh                  -> Lint workbench checks passed
bash -n tests/*.sh                            -> pass
git diff --check                              -> pass
NVIM_BIN=/home/barry/.local/bin/nvim \
  NVIM_DATA_HOME=/home/barry/.local/share \
  bash tests/smoke.sh                          -> pass
```

冒烟测试使用 `MARS_SMOKE=1` 禁止 Treesitter 下载，并使用已经同步的插件数据目录；
配置目录、状态目录和缓存目录仍为临时目录。`greeter.lua` 已改为从自身文件定位，
因此临时 XDG 配置不会改变本地插件路径。`leap.nvim` 固定在最后一个包含实现的提交，
避免当前上游归档分支在启动时加载失败。

`tests/ai_helper.sh` 还验证了 `#{diff}`、`#{terminal}`、`#{quickfix}` 标签、搜索结果
quickfix 跳转、UTF-8 输入清理和跨项目会话拒绝。`vim.o.mouse` 为空，终端和编辑器操作
均使用键盘映射。

`nvim-dap` 的 Python adapter 已用项目解释器和 debugpy 1.8.21 启动 DAP initialize；
GDB 15.1 的 `--interpreter=dap` 也返回 initialize/initialized 响应。缺少 adapter 时
只发出 warning，不阻塞配置加载。

Codex 启动器会先检查 `$CODEX_HOME/sessions` 中当前项目的 `session_meta.payload.cwd`：
有匹配历史才执行 `resume --last`，resume 错误码原样返回；无历史或设置
`MARS_CODEX_NEW=1` 时才启动新会话。

原生 `:CodeCompanionCLI` 已由 Mars wrapper 接管：它先执行项目根/会话校验，拒绝
跨项目复用，并把视觉命令的 range 传给 CodeCompanion。项目外当前 buffer 的自动
`#{buffer}`、`#{this}`、`#{selection}` 和诊断标签会被省略；显式用户文件引用不受影响。
Markdown lint 在 `markdownlint`/`markdownlint-cli2` 不存在时跳过，不阻塞编辑。

`MARS_SMOKE=1` 同时关闭 Treesitter 自动下载和 lazy 缺失插件安装；插件缓存不完整
时会快速失败而不会隐式联网。当前 smoke 使用已同步的用户级插件缓存，真实工作流
仍通过 `:Lazy sync` 完成首次安装。

## 合并探针

实现分支已 fast-forward 合并到 `/home/barry/.config/mars.nvim` 的 `main`，合并
结果先为 `c340ef9`，随后补充默认 session 目录修复，当前 `main` HEAD 为 `9bf0794`；
隔离 worktree 和 feature branch 随后清理。使用临时 `CODEX_HOME`
启动合并后的真实配置并调用 `require('mars.ai').toggle()`：成功创建 live
`codecompanion_cli` terminal，项目 root marker 正确，`require('mars.ai').reset_session()`
完成生命周期清理，结果为 `MERGED_PANEL_PASS`。

post-resume 修复后再次执行全部 `tests/*.sh`、静态检查、smoke 和面板探针，分别得到
`POST_RESUME_ALL_TESTS_PASS`、`POST_RESUME_STATIC_PASS`、`POST_RESUME_SMOKE_PASS` 和
`POST_RESUME_PANEL_PASS`。启动器默认读取 `$HOME/.codex/sessions`，并覆盖了未设置
`CODEX_HOME` 的回归场景。

## ACP 探测

在临时 `CODEX_HOME` 和 `NO_BROWSER=1` 环境中向 `codex-acp` 发送 newline-delimited
JSON `initialize` 请求，返回 `protocolVersion: 1`、Codex agent 信息和 session 能力。
初始化成功，但尚未把完整的 session restore、权限请求、文件读写、取消和终端事件
映射到 Mars.nvim，因此 ACP 没有替换日常 CLI provider。

## 安全检查

- Lua 配置未发现 `OPENAI_API_KEY`、`CODEX_API_KEY`、`api_key`、`access_token`、
  `refresh_token` 或 `auth_token` 赋值。
- Lua 配置未传入 `--dangerously-bypass-approvals-and-sandbox`。
- Codex 认证继续由 Codex 自己管理；没有把凭据写入仓库或补全文件。
- 搜索上下文排除 `.env*`、密钥/证书、`.git`、构建产物、`dist` 和 `node_modules`，
  并限制结果字节数和行数。

## 已知限制

- 系统 `apt` 安装因当前 sudo 需要交互密码未执行；clangd、clang-format、debugpy、
  fd 和 lazygit 使用用户级安装，功能验证通过。
- 首次使用前仍需在 Neovim 中运行 `:Lazy sync`；冒烟测试在插件数据目录缺失时会快速
  失败并提示该前置条件，不会自动联网克隆。
- 真实 Codex 模型请求、审批交互和修改接受/拒绝需要在用户自己的项目中手动验收，
  自动测试使用 mock 以避免发送请求或改变项目文件。
- VectorCode、Overseer、code-preview.nvim 和 ACP 默认后端属于后续阶段，当前未强制
  引入。
