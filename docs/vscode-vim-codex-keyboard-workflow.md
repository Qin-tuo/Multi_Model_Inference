# VS Code + VSCodeVim + Codex 全键盘工作流

本文记录当前主机实际配置的无鼠标工作流，目标是在代码编辑器、终端、文件资源管理器和 Codex 之间完成键盘切换、文件操作、对话导航、代码块复制与常用开发操作。

## 1. 当前环境与配置范围

当前环境：

- Linux x64；
- VS Code 1.134.0；
- VSCodeVim 1.32.4；
- OpenAI Codex 扩展 26.825.51511；
- Codex Webview 本地键盘增强已安装。

配置来源：

| 配置 | 路径 |
| --- | --- |
| VS Code 用户快捷键 | `~/.config/Code/User/keybindings.json` |
| VS Code 用户设置 | `~/.config/Code/User/settings.json` |
| Codex Webview 增强源码 | `~/.local/share/codex-webview-keyboard-tools/codex-keyboard-tools.js` |
| Codex Webview 增强测试 | `~/.local/share/codex-webview-keyboard-tools/test.js` |
| Codex 增强安装器 | `~/.local/bin/patch-codex-webview-keyboard-tools` |

## 2. 核心视图切换

| 快捷键 | 功能 |
| --- | --- |
| `Alt+1` | 聚焦第一个代码编辑器组 |
| `Alt+2` | 聚焦集成终端 |
| `Alt+3` | 打开并聚焦 Codex 侧栏 |
| `Alt+4` | 聚焦文件资源管理器 |
| `Ctrl+Alt+B` | 显示或隐藏右侧辅助栏 |

说明：

- 默认的 `Ctrl+1` 已取消，统一使用 `Alt+1` 返回编辑器。
- 四个 Alt 视图命令已加入 `terminal.integrated.commandsToSkipShell`，在终端中按下时由 VS Code 处理，不会作为 Meta 序列发送给 shell。

## 3. 文件资源管理器

以下单字母键只在文件资源管理器获得焦点且没有输入框活动时生效。

| 快捷键 | 功能 |
| --- | --- |
| `j` / `k` | 选择下一项/上一项 |
| `h` / `l` | 折叠目录/展开目录 |
| `Enter` | 打开选中文件 |
| `Ctrl+Enter` | 在侧边编辑器组打开 |
| `a` | 在当前目录新建文件 |
| `Shift+A` | 在当前目录新建文件夹 |
| `r` | 重命名文件或目录 |
| `d d` | 删除到回收站 |
| `y` | 复制文件或目录 |
| `x` | 剪切文件或目录 |
| `p` | 粘贴文件或目录 |
| `/` | 查找或过滤文件树 |
| `Shift+R` | 刷新资源管理器 |
| `Shift+H` | 折叠所有目录 |
| `Shift+F10` | 打开选中项的上下文菜单 |

注意：当前设置包含 `"explorer.confirmDelete": false`，因此 `d d` 会直接把选中项移入回收站，不显示确认对话框。

## 4. Codex 会话操作

| 快捷键 | 功能 |
| --- | --- |
| `Ctrl+Alt+S` | 打开 Codex 会话搜索与切换器 |
| `Ctrl+[` | 返回会话列表或上一层 |
| `Ctrl+N` | 开启新会话 |

选择会话：

```text
Alt+3          进入 Codex
Ctrl+Alt+S     打开会话搜索
输入名称       过滤会话
Up / Down      选择会话
Enter          打开会话
```

`Ctrl+Alt+S` 和 `Ctrl+[` 由 Webview 增强模块直接向 Codex 本地消息总线发送 `chat-search-command-menu` 和 `navigate-back` 消息，不依赖 Codex 桌面版专用 keymap。

## 5. Codex 对话滚动

| 快捷键 | 功能 |
| --- | --- |
| `Alt+J` | 向下滚动 160px |
| `Alt+K` | 向上滚动 160px |
| `Alt+D` | 向下翻动当前可视高度的 80% |
| `Alt+U` | 向上翻动当前可视高度的 80% |
| `Alt+Home` | 跳到最早消息 |
| `Alt+End` | 跳到最新消息 |
| `Alt+G` | 跳到最新消息，作为 `Alt+End` 的别名 |

Codex 时间线使用反向滚动布局：最新消息位置为 `0`，最早消息位置为 `clientHeight - scrollHeight`。增强模块已按这个模型处理 Home、End 和滚动方向。

## 6. Codex 单个代码块操作

| 快捷键 | 功能 |
| --- | --- |
| `Alt+N` | 选择下一个代码块 |
| `Alt+P` | 选择上一个代码块 |
| `Alt+C` | 复制当前选中的单个代码块 |

选择规则：

1. 优先确定当前获得焦点的可见 Codex 时间线。
2. 只查找该时间线中可见且已渲染的代码块。
3. 排除 `hidden`、`aria-hidden`、`inert` 和零尺寸代码块。
4. 第一次选择时使用距离视口中心最近的代码块。
5. 后续 `Alt+N/P` 按当前 DOM 顺序循环选择。

选中后的表现：

- 目标代码块显示蓝色轮廓；
- 页面顶部短暂显示 `Code block X/Y`；
- `Alt+C` 只复制代码块原始文本，不包含语言标签、按钮或回复说明；
- 未预先选择时，`Alt+C` 自动复制视口中心最近的代码块。

页面随动采用最小移动策略：代码块已经可见时不移动；超出可视区域时只滚动必要距离；虚拟化重新布局后在下一帧校正一次。

Codex 会虚拟化较早的消息。若目标代码块尚未渲染，先用 `Alt+K` 或 `Alt+U` 向上滚动，等内容进入 DOM 后再使用 `Alt+N/P`。

## 7. VSCodeVim 编辑操作

| 快捷键 | 功能 |
| --- | --- |
| `Esc` | 返回 Normal 模式 |
| `h` / `j` / `k` / `l` | 左/下/上/右移动 |
| `w` / `b` | 下一个/上一个单词 |
| `0` / `$` | 行首/行尾 |
| `gg` / `G` | 文件开头/文件末尾 |
| `i` / `a` / `o` | 插入/追加/新建下一行 |
| `dd` | 删除当前行 |
| `yy` | 复制当前行 |
| `p` | 粘贴 |
| `u` / `Ctrl+R` | 撤销/重做 |
| `/` | 搜索 |
| `n` / `N` | 下一个/上一个搜索结果 |
| `gd` | 跳转到定义 |
| `gh` | 显示类型、错误等悬浮信息 |
| `Ctrl+S` | 保存文件 |

## 8. 常用 VS Code 原生快捷键

| 快捷键 | 功能 |
| --- | --- |
| `Ctrl+Shift+P` / `F1` | 命令面板 |
| `Ctrl+Shift+E` | 显示或聚焦资源管理器 |
| `Ctrl+K`，然后 `R` | 在资源管理器中定位当前文件 |
| `Ctrl+Shift+F` | 全局搜索 |
| `Ctrl+Shift+G` | 源代码管理 |
| `Ctrl+Shift+M` | 问题面板 |
| `F8` / `Shift+F8` | 下一个/上一个问题 |
| `` Ctrl+` `` | 显示或隐藏集成终端 |
| `` Ctrl+Shift+` `` | 新建集成终端 |

VSCodeVim 默认会接管部分 Ctrl 组合键。例如编辑器 Normal 模式中的 `Ctrl+B` 是向上翻页，不应把它当作 VS Code 侧栏切换键；本配置使用 `Alt+1..4` 规避这类冲突。

## 9. 推荐日常工作流

### 9.1 浏览并编辑文件

```text
Alt+4          进入文件树
j / k          选择文件
h / l          折叠或展开目录
Enter          打开文件
Alt+1          聚焦编辑器
Esc            进入 Vim Normal 模式
```

### 9.2 编辑后运行命令

```text
Ctrl+S         保存
Alt+2          进入终端
运行构建或测试命令
Alt+1          返回编辑器
```

### 9.3 向 Codex 提问并返回代码

```text
Alt+3          进入 Codex
输入问题
Enter          发送
Alt+1          返回编辑器
```

### 9.4 从 Codex 复制单个代码块

```text
Alt+3          进入 Codex
Alt+N / Alt+P  选择目标代码块
Alt+C          复制代码块
Alt+1          返回编辑器
p               使用 Vim 粘贴
```

### 9.5 切换或新建 Codex 会话

```text
Ctrl+Alt+S     搜索并切换会话
Ctrl+[         返回会话列表
Ctrl+N         新建会话
```

## 10. Codex 增强维护

Codex 扩展升级后，新版本目录会覆盖 Webview 入口。重新执行：

```bash
patch-codex-webview-keyboard-tools
```

然后重载 VS Code：

```text
Ctrl+Shift+P -> Developer: Reload Window
```

运行测试：

```bash
node --test ~/.local/share/codex-webview-keyboard-tools/test.js
```

完全恢复官方 Webview：

```bash
patch-codex-webview-keyboard-tools --restore
```

安装器会保留原始 `webview/index.html` 备份，重复安装不会重复插入脚本标签。

## 11. 故障排查

### 快捷键完全没有响应

1. 执行 `Developer: Reload Window`。
2. 确认当前 Codex 扩展版本仍已打补丁：

```bash
patch-codex-webview-keyboard-tools
```

3. 再次重载窗口。

### 在终端按 Alt 视图键没有切换

检查 `~/.config/Code/User/settings.json` 中的 `terminal.integrated.commandsToSkipShell` 是否包含：

```json
[
  "workbench.action.focusFirstEditorGroup",
  "terminal.focus",
  "chatgpt.openSidebar",
  "workbench.files.action.focusFilesExplorer"
]
```

### 代码块蓝框正确但页面没有跟随

确认已安装最新增强并重载窗口。当前实现不会强制平滑居中，而是执行最小位移并在下一帧校正，避免与 Codex 反向虚拟化时间线冲突。

### `Ctrl+Alt+S` 无响应

确认增强模块是最新版本并重载窗口。该按键由增强模块直接投递 Codex 本地 `chat-search-command-menu` 消息，不需要 `~/.codex/keybindings.json`。

## 12. 作用域与限制

- Codex 滚动与代码块快捷键只在 Codex Webview 内生效。
- 文件树字母快捷键只在文件资源管理器获得焦点且没有输入框活动时生效。
- 当前增强依赖 Codex 的 `data-pip-anchor-host`、`data-markdown-copy` 和 `data-markdown-copy-text` DOM 标记。
- OpenAI Codex 扩展更新可能改变这些标记；安装器会在入口结构不匹配时失败，而不是模糊修改未知版本。
- 这是本机用户级增强，不属于项目源代码，也不应作为项目运行依赖。
