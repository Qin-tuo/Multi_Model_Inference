# AI-Native CLI 编辑器设计规格

## 1. 背景

目标是为个人开发工作流打造一个以 Neovim 为核心、完全键盘驱动、由
Codex 协作的 CLI 编辑器。主要工作环境是 Ubuntu 24.04 x86_64，开发语言
为 Python、C++ 和 CUDA，项目以 Git 和 CMake 为基础。

当前已有一套 Mars.nvim 配置，包含 lazy.nvim、Snacks、Telescope、Neogit、
gitsigns、LSP、补全和调试的基础结构。当前 Neovim 为 0.9.5，配置和部分
插件的主线版本已经要求 0.10、0.11 或更高版本，因此升级编辑器版本是
实现前置条件。

## 2. 目标

### 2.1 产品目标

构建一个只服务个人的 AI-native CLI 开发工作台，覆盖以下闭环：

```text
表达目标 -> AI 理解项目 -> 补全需求和计划 -> 修改代码 -> 审查 diff
        -> 构建/测试/调试 -> 检查 Git 状态 -> 人工确认后提交
```

### 2.2 体验目标

- 常见开发操作不依赖鼠标。
- 常驻 AI 面板保持项目会话和上下文；命令与快捷键用于快速触发动作。
- AI 能看到当前文件、选区、诊断、终端输出、Git diff 和相关文档。
- 用户只描述意图时，AI 主动补全约束、边界情况、依赖、测试和验收标准。
- 文件修改先进入可审查的 diff 流程，危险 Git 操作必须明确确认。
- 结果可以跳转到文件和行号，计划可以转换为待办，测试失败可以继续交给 AI 修复。

### 2.3 技术目标

- Python：语言服务、补全、类型诊断、Ruff 检查/格式化、pytest 和 debugpy。
- C++/CUDA：clangd、CMake 编译数据库、clang-format、GDB/DAP，以及 CUDA
  工具链和已有的 `compile_commands.json`。
- Git：状态、diff、hunk 暂存、分支、历史、冲突和提交前检查。
- 文档和代码：快速文本搜索优先，语义索引作为后续增强。

## 3. 非目标

- 第一版不做跨平台发行版、多人协作、云端服务或通用插件市场。
- 不重写 Codex 模型、Codex 权限系统或 Codex CLI 的 TUI。
- 不在第一版自研完整向量数据库、代码解析器或 LSP 实现。
- 不把所有开发命令自动化为无需确认的后台操作。
- 不为了追求 IDE 外观加入与工作流无关的 UI 装饰。

## 4. 总体架构

```text
                         +-----------------------------+
                         |        Codex 后端            |
                         | codex TUI / app-server / ACP |
                         +--------------+--------------+
                                        |
                         terminal process or ACP stream
                                        |
+------------------+          +--------v--------+          +------------------+
| Python/C++/CUDA  | <-------> | CodeCompanion   | <------> | 项目上下文层      |
| 编辑、LSP、DAP   |           | 常驻聊天面板     |          | rg/Git/Markdown   |
+------------------+           +--------+--------+          | VectorCode(后续)  |
                                        |                   +------------------+
                               +--------v--------+
                               | Neovim / Snacks |
                               | 窗口、终端、输入 |
                               +-----------------+
```

职责边界：

- Neovim 负责编辑器状态、窗口布局、键盘映射、LSP/DAP 和本地文件缓冲区。
- CodeCompanion 负责聊天缓冲区、编辑器上下文、工具、Slash Commands、diff
  交互和工作流编排；它的 CLI provider 也可以托管真实 Codex TUI。
- 真实 `codex` TUI 负责完整的交互、命令执行和原生审批；`codex-acp`/Codex
  app-server 提供可选的结构化会话和事件通道。
- `rg`、Git、CMake、编译器、测试框架和调试器负责确定性的本地操作。
- 自定义桥接层只补足项目上下文、命令封装、个人快捷键和安全策略，不复制已有插件的核心能力。

## 5. 插件评估与选型

评估日期：2026-09-03。

| 能力 | 候选 | 决策 | 原因 |
| --- | --- | --- | --- |
| 常驻 AI 面板和工作流 | `olimorris/codecompanion.nvim` | 采用 | 支持常驻 chat buffer、ACP、上下文、Slash Commands、工具、代码审查和 diff |
| Codex 连接 | CodeCompanion CLI provider + `codex` | MVP 采用 | 由 Neovim 管理持久终端，保留 Codex 原生交互、命令执行和审批语义 |
| 结构化 Codex 连接 | `codex-acp` + Codex app-server | 验证后启用 | 提供结构化会话、权限、文件操作和事件；当前 CodeCompanion ACP 端对 terminal 能力有限 |
| 终端和布局 | `folke/snacks.nvim` | 复用 | 已有，提供 terminal、picker、input、窗口和 lazygit 集成 |
| 轻量 Codex 终端封装 | `leejh903/codex.nvim` | 参考/备用 | 能同步 Codex `/ide` IPC，但项目规模较小，不作为核心依赖 |
| 简单 Codex 终端封装 | `dimfeld/codex.nvim` | 不采用 | 仅打开终端和插入当前文件引用，无法覆盖完整工作流 |
| 另一套 AI 编辑器 | `avante.nvim` | 暂不采用 | 功能重叠，当前要求更高 Neovim 版本，增加迁移和维护面 |
| AI 修改前原生 diff | `Cannon07/code-preview.nvim` | 第二阶段评估 | 支持 Codex hooks 和 Neovim diff，但依赖 Codex hooks 配置，先验证稳定性 |
| 文本文件和代码搜索 | Snacks picker / Telescope / `rg` | 复用并统一入口 | 已有插件和 `rg`，足以覆盖文件、符号、全文和 Markdown 搜索 |
| 语义项目检索 | `Davidyz/VectorCode` | 第二阶段评估 | 可通过 CodeCompanion 工具查询仓库，但当前为 beta，需先验证索引和隐私边界 |
| Git 行级操作 | `lewis6991/gitsigns.nvim` | 复用 | 提供 hunk 导航、暂存、重置、预览和 blame |
| Git 状态与提交 | Neogit + diffview | 复用 | 已有 Magit 风格状态界面和 diff 查看能力 |
| Git 全流程 TUI | lazygit + Snacks | 补充 | 对交互式 rebase、冲突和复杂历史更高效；需安装 `lazygit` 二进制 |
| 构建和任务 | CMake Tools 或 Overseer | 第二阶段评估 | 先用项目脚本和终端验证；任务稳定后再统一任务入口 |

CodeCompanion 的 CLI provider 可以直接运行 `codex`，而 ACP 模式另外要求
安装 `codex-acp`。当前 CodeCompanion 主线要求 Neovim 0.11+；当前
`nvim-lspconfig` 主线要求 0.11.3+，Treesitter 主线要求更高版本，因此
版本锁定和升级顺序必须写入实施计划。ACP 模式只有在会话恢复、权限、文件
修改和命令输出的端到端试验通过后，才提升为默认后端。

## 6. 常驻面板交互设计

### 6.1 面板状态

- 面板默认在右侧垂直分栏，占编辑器宽度约 35% 至 42%；MVP 面板内部是
  由 CodeCompanion 管理的持久 Codex 终端。
- 关闭窗口只隐藏面板，不终止 Codex 会话。
- 可通过一个全局快捷键在编辑器和面板之间切换焦点。
- 同一项目保持一个默认会话；需要时支持新建、恢复、继续和分叉会话。
  终端模式使用 Codex 自身的会话命令，ACP 模式使用 app-server 的 session API。
- 面板显示当前 adapter、模型、项目根目录、处理状态和待确认操作。

### 6.2 输入模型

普通文本直接作为任务描述，例如：

```text
为这个 CUDA 推理接口增加批处理，并补充失败场景测试
```

结构化入口用于高频动作：

```text
/ask       询问项目问题
/search    搜索代码和 Markdown
/plan      补全需求并生成计划
/edit      生成修改
/test      构建或运行测试
/review    检查当前 diff
/git       解释或准备 Git 操作
```

这些入口优先实现为 CodeCompanion 的 Slash Commands、prompt library 和
个人快捷键，而不是另造一套聊天协议。

### 6.3 上下文快捷入口

面板必须支持通过补全选择并附加以下上下文：

- 当前文件、当前选区和多个文件引用。
- 当前行或当前缓冲区的 LSP 诊断。
- 最近一次构建、测试或终端输出。
- 当前分支、工作区状态和 staged/unstaged diff。
- `README.md`、`docs/`、设计规格和项目规则文件。
- 通过 `rg` 返回的文件路径、匹配行和短上下文。

面板中要显示本轮实际附加的文件和命令来源，避免上下文不可见。

## 7. 项目搜索与 Markdown 策略

采用两级搜索：

1. **确定性搜索（MVP）**：使用 `rg`，遵守 `.gitignore`，提供文件名、全文、
   Markdown 专用和 Git 跟踪文件搜索；结果进入 quickfix 或直接跳转。
2. **语义搜索（后续）**：验证 VectorCode 的本地索引和 CodeCompanion 工具集成，
   仅索引明确允许的源码和文档目录，排除密钥、缓存、构建产物和大二进制。

AI 的默认项目上下文顺序为：当前文件/选区 -> 诊断和终端输出 -> Git diff ->
确定性搜索结果 -> 语义检索结果。语义检索不能替代可复现的文本搜索，也不能
在没有来源路径的情况下把结果直接当作事实。

## 8. Git 工作流

日常 Git 操作分为三层：

- 编辑中：gitsigns 显示变化，支持 hunk 导航、预览、暂存和撤销暂存。
- 审查中：Neogit/diffview 或 lazygit 查看完整状态、历史、冲突和提交范围。
- AI 协作中：CodeCompanion 使用 `#{diff}` 或自定义 Git 上下文生成审查意见、
  提交说明和提交前检查清单。

以下动作必须人工确认：`commit`、`push`、`reset --hard`、批量删除、强制重写历史
和任何超出当前项目根目录的写操作。AI 可以准备命令和说明，但不能静默执行这些动作。

## 9. Python、C++ 和 CUDA 工具链

### Python

- LSP：Pyright 或 BasedPyright，按项目虚拟环境解析解释器。
- 检查和格式化：Ruff；避免同时启用多个互相覆盖的格式化器。
- 测试：pytest；失败输出进入终端上下文和 quickfix。
- 调试：debugpy + nvim-dap；使用项目解释器而不是系统 Python 2.7 命令。

### C++/CUDA

- LSP：clangd，优先读取项目生成的 `compile_commands.json`。
- 格式化：clang-format，格式规则以项目 `.clang-format` 为准。
- 构建：CMake、Make/Ninja 和项目现有脚本；构建输出可解析为 quickfix。
- 调试：GDB/DAP；必要时再加入 codelldb，不为替代 GDB 而重复安装。
- CUDA：保留 `nvcc`、CUDA 环境变量和目标平台编译参数；clangd 配置不能覆盖
  CUDA 编译数据库中的真实 include、宏和架构选项。

## 10. 版本与安装策略

实现前先将 Neovim 升到稳定的 0.11.x（或当时已验证的更高稳定版），并记录：

- `nvim --version`、lazy.nvim 锁文件和每个关键插件的版本/commit。
- Codex CLI、`codex-acp` 和 app-server 的版本。
- Python、Node、clangd、Ruff、Pyright、debugpy、GDB、CMake 和 CUDA 版本。

不在 Neovim 0.9.5 上强行追随当前插件主线。升级后重新验证：

1. 现有 Mars.nvim 基础启动和插件同步。
2. CodeCompanion 独立 chat buffer。
3. Codex ACP 登录、会话恢复、文件读取、写入审批和中断。
4. Python/C++ 文件的 LSP、补全、诊断、格式化和调试。
5. Git diff、Markdown 搜索和项目根目录识别。

## 11. 安全和可恢复性

- 凭据只由 Codex 自己的认证机制或环境管理，不写入 Lua、Markdown 或 Git。
- 默认只允许访问当前项目根目录及显式加入的目录。
- `.env`、密钥文件、SSH 凭据、令牌和被忽略的敏感文件不自动进入上下文。
- AI 修改必须可以查看 diff、接受、拒绝和撤销。
- shell 命令显示完整命令、工作目录和权限模式；危险命令需要确认。
- 面板会话、日志和索引存放在本地数据目录，不提交到项目仓库。
- 发生插件或 Codex 故障时，仍能直接使用原生编辑器、终端和 Git 命令。

## 12. 分阶段交付

### MVP-0：基础可用性

- 升级并锁定 Neovim/插件版本。
- 安装 Python/C++ 必需外部工具。
- 保证现有配置无错误启动，补齐 Python、C++、CUDA 文件类型和基本 LSP。

### MVP-1：常驻 Codex 面板

- CodeCompanion 常驻面板通过 CLI provider 接入真实 `codex` TUI。
- 项目根目录启动、会话保持、恢复、焦点切换和中断。
- 当前文件、选区、诊断、终端和 Git diff 上下文。
- `/ask`、`/search`、`/plan`、`/review` 基础入口。

### MVP-2：修改、验证和 Git 闭环

- 文件修改的 diff 审查和接受/拒绝。
- `/edit`、`/test`、`/git` 工作流。
- CMake/pytest 任务、quickfix 错误回传和 AI 迭代修复。
- Neogit/diffview/lazygit 的统一快捷键和确认策略。

### Phase-2：知识和自动化增强

- 验证并按需启用 CodeCompanion 的 `codex-acp` 后端，比较它与 CLI provider
  在会话、权限、事件和上下文同步上的差异。
- VectorCode 语义索引和 CodeCompanion 工具集成。
- Codex hooks 与 code-preview.nvim 的稳定性评估。
- 任务编排、自动生成项目状态摘要、提交前检查清单和本地会话归档。

## 13. 验收标准

设计实现完成后，以下场景必须可以仅用键盘完成：

1. 在项目根目录启动 Neovim，打开并聚焦常驻 Codex 面板。
2. 询问一个涉及 README、源码和当前文件的问题，AI 能列出实际使用的来源。
3. 搜索一个 Markdown 主题并跳转到匹配行，再把结果交给 Codex 分析。
4. 选择一段 Python 或 C++ 代码，请 AI 解释、规划修改并显示 diff。
5. 接受修改后运行 pytest 或 CMake 构建，失败信息能回到面板继续修复。
6. 查看当前 Git diff，让 AI 生成审查意见和提交说明，但不会自动提交。
7. 隐藏并重新打开面板后，会话仍在；退出 Neovim 后能按项目恢复历史会话。
8. 在缺少某个外部工具或 Codex 服务不可用时，编辑、搜索和普通 Git 操作仍可用。

验证记录必须包含版本清单、关键命令输出、失败处理结果和已知限制；不以一次
成功启动作为完整验收。

## 14. 关键假设

- 第一版只服务当前用户和当前 Ubuntu 工作站。
- Codex CLI 继续作为主要 AI 后端，未来可以通过 ACP 替换或增加其他 agent。
- 用户接受先升级 Neovim，再使用 CodeCompanion 当前支持的插件版本。
- 项目代码和文档主要保存在本地，语义索引可以使用本地数据库和本地缓存。
- 所有自动化都以可审查、可撤销和明确权限为优先级。

## 15. 评估依据

- [CodeCompanion.nvim](https://github.com/olimorris/codecompanion.nvim) 与其
  [安装文档](https://codecompanion.olimorris.dev/installation)：chat buffer、CLI、
  ACP、上下文、工具和 diff。
- [CodeCompanion ACP 配置](https://codecompanion.olimorris.dev/configuration/adapters-acp)：
  Codex 需要 `codex-acp`，以及 adapter 和认证配置。
- [OpenAI Codex CLI](https://github.com/openai/codex) 与
  [app-server 文档](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)：
  CLI、会话和 JSONL app-server 接口。
- [Snacks.nvim](https://github.com/folke/snacks.nvim)：terminal、picker 和 lazygit
  集成，要求 Neovim 0.9.4 或更高。
- [leejh903/codex.nvim](https://github.com/leejh903/codex.nvim)：Codex 原生 `/ide`
  IPC 的参考实现。
- [code-preview.nvim](https://github.com/Cannon07/code-preview.nvim)：Codex hooks
  和原生 Neovim diff 的后续评估对象。
- [VectorCode](https://github.com/Davidyz/VectorCode)：仓库索引、语义查询和
  CodeCompanion 工具集成的后续评估对象。
- [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig)、
  [Ruff Neovim 设置](https://docs.astral.sh/ruff/editors/setup/) 和
  [Neovim 安装文档](https://neovim.io/doc/install/)：语言服务与版本约束。
