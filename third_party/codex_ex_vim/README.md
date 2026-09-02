# Codex Webview + Vim 键盘增强

此目录保存当前主机上为 VS Code Codex 扩展增加的键盘层，供主仓库进行版本管理。

## 当前保存位置

实际运行中的官方扩展是：

```text
/home/barry/.vscode/extensions/openai.chatgpt-26.825.51511-linux-x64
```

可维护的增强源码原先保存在：

```text
/home/barry/.local/share/codex-webview-keyboard-tools/codex-keyboard-tools.js
/home/barry/.local/share/codex-webview-keyboard-tools/test.js
/home/barry/.local/bin/patch-codex-webview-keyboard-tools
```

本目录是上述三份文件的 Git 副本。功能源码保持字节一致；为让仓库副本可以独立运行，
副本测试使用相对路径，副本安装器优先读取同目录源码。当前安装器实际改动官方扩展中的两个文件：

```text
webview/index.html                    # 注入增强模块的 script 标签
webview/codex-keyboard-tools.js      # 增强模块副本
```

安装器还会在官方扩展的 `webview/index.html.codex-keyboard-tools.original`
保存干净入口，便于恢复。官方扩展本体及其 `assets`、二进制文件没有复制到本仓库。

## 文件说明

- `codex-keyboard-tools.js`：Codex Webview 的键盘增强；支持会话导航、时间线滚动和单个代码块选择/复制。
- `test.js`：Node.js 内置测试，覆盖选择、滚动、复制、消息总线和补丁幂等性。
- `patch-codex-webview-keyboard-tools`：检测当前 `openai.chatgpt` 版本并可重复安装或恢复补丁。

## 使用

在仓库根目录执行测试：

```bash
node --test third_party/codex_ex_vim/test.js
```

安装到当前用户的 Codex 扩展：

```bash
node third_party/codex_ex_vim/patch-codex-webview-keyboard-tools
```

恢复官方 Webview：

```bash
node third_party/codex_ex_vim/patch-codex-webview-keyboard-tools --restore
```

仓库副本的安装器默认读取同目录下的 `codex-keyboard-tools.js`；如果从旧的
`~/.local/bin` 安装器运行，则仍回退到 `~/.local/share/codex-webview-keyboard-tools`。
也可以通过 `CODEX_KEYBOARD_TOOLS_SOURCE` 指定源码路径。

Codex 扩展升级后，重新运行安装命令并在 VS Code 中执行
`Developer: Reload Window`。升级会覆盖官方扩展目录中的补丁，但不会删除本目录的
Git 源码。

当前验证基线：`openai.chatgpt@26.825.51511`，增强源码与已安装的
`webview/codex-keyboard-tools.js` 内容一致。
