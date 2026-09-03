# AI-Native CLI Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在个人 Ubuntu 工作站上把 Mars.nvim 变成一个由 Neovim 管理、以 Codex CLI 为后端、完全键盘驱动的 AI 开发工作台。

**Architecture:** Neovim 承载编辑、LSP、DAP、Git 和窗口状态；CodeCompanion 提供常驻 CLI 面板和编辑器上下文；真实 `codex` TUI 负责 AI 会话、命令执行和审批。确定性搜索由 `rg`/Snacks picker 提供，语义检索和 ACP 原生面板留到验证阶段。

**Tech Stack:** Neovim 0.12.5、lazy.nvim、CodeCompanion.nvim、Snacks.nvim、Codex CLI 0.152.x、Python/Pyright/Ruff/pytest/debugpy、C++/CUDA/clangd/CMake/GDB、Git/Neogit/gitsigns/Fugitive。

---

## 工作目录与文件边界

实现分支位于：

```text
/home/barry/AI_Infra/Orin_NX_Multi_Model_Inference/.worktrees/mars-ai-editor
```

它对应 Mars.nvim 仓库的 `feature/mars-ai-editor` 分支。实现完成并通过验证后，
将该分支 fast-forward 合并回 `/home/barry/.config/mars.nvim` 的 `main`。

计划涉及的文件：

- Modify: `nvim/init.lua`，注册新插件、自动重载和全局工作流。
- Modify: `nvim/lua/mars/options.lua`，版本目标、项目文件自动重载和语言缩进。
- Modify: `nvim/lua/mars/health.lua`，检查版本和个人工作台外部依赖。
- Modify: `nvim/lua/mars/plugins/ai.lua`，停用与 Codex 快捷键冲突的旧 AI 入口。
- Modify: `nvim/lua/mars/plugins/lsp.lua`，使用现代 LSP API 配置 Pyright 和 clangd。
- Modify: `nvim/lua/mars/plugins/lint.lua`，启用 Python Ruff 诊断。
- Modify: `nvim/init.lua` 中的 Conform 和 Treesitter 配置，加入 Python/C++/CUDA。
- Modify: `nvim/init.lua` 中的 Supermaven 依赖，避免第二个默认 AI 入口抢占补全。
- Create: `nvim/lua/mars/plugins/codecompanion.lua`，常驻 Codex CLI 面板和 AI 命令。
- Create: `nvim/lua/mars/ai.lua`，项目根目录、Markdown 搜索和上下文发送辅助函数。
- Create: `nvim/lua/mars/plugins/git-workbench.lua`，Fugitive 和 Snacks lazygit 入口。
- Create: `tests/smoke.sh`，配置、命令和外部工具的无网络冒烟测试。
- Create: `tests/context.sh`，临时 Git 仓库中的 Markdown 搜索集成测试。
- Modify: `README.md`、`README_zh.md`，记录安装顺序、快捷键和故障排查。

不修改当前主项目中已有的学习记录、模型代码或未提交文件。

### Task 1: 建立失败冒烟测试和干净基线

**Files:**
- Create: `/home/barry/AI_Infra/Orin_NX_Multi_Model_Inference/.worktrees/mars-ai-editor/tests/smoke.sh`
- Create: `/home/barry/AI_Infra/Orin_NX_Multi_Model_Inference/.worktrees/mars-ai-editor/tests/context.sh`

- [ ] **Step 1: 写出失败冒烟测试**

创建 `tests/smoke.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nvim_bin="${NVIM_BIN:-nvim}"

command -v "$nvim_bin" >/dev/null
command -v codex >/dev/null
command -v rg >/dev/null

"$nvim_bin" --headless -u "$repo_root/nvim/init.lua" -i NONE \
  -c 'lua assert(vim.fn.has("nvim-0.12") == 1, "Neovim 0.12 is required")' \
  -c 'lua assert(vim.fn.exists(":CodeCompanionCLI") == 2, "CodeCompanionCLI is missing")' \
  -c 'lua assert(vim.fn.exists(":AI") == 2, ":AI is missing")' \
  -c 'lua assert(vim.fn.exists(":Git") == 2, ":Git is missing")' \
  -c 'qa!'
```

创建 `tests/context.sh`，验证 Markdown 搜索只返回允许的文件：

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT

git -C "$fixture" init -q
printf '%s\n' '# target phrase' > "$fixture/README.md"
printf '%s\n' '# target phrase' > "$fixture/notes.md"
printf '%s\n' 'target phrase' > "$fixture/source.cpp"
printf '%s\n' 'target phrase' > "$fixture/.secret.md"
printf '%s\n' '.secret.md' > "$fixture/.gitignore"

output="$(rg --color=never --line-number --column --no-heading \
  --glob '*.md' --glob '!.git/**' --glob '!.secret.md' \
  'target phrase' "$fixture")"

grep -q 'README.md' <<<"$output"
grep -q 'notes.md' <<<"$output"
! grep -q 'source.cpp' <<<"$output"
! grep -q '.secret.md' <<<"$output"
```

- [ ] **Step 2: 运行基线测试并记录失败原因**

Run:

```bash
cd /home/barry/AI_Infra/Orin_NX_Multi_Model_Inference/.worktrees/mars-ai-editor
./tests/smoke.sh
```

Expected before implementation: failure because the host currently has Neovim
0.9.5 and the `:CodeCompanionCLI`/`:AI` commands do not exist. `tests/context.sh`
should pass because it only depends on the existing `rg` binary.

- [ ] **Step 3: Commit the test baseline**

```bash
git add tests/smoke.sh tests/context.sh
git commit -m "test: add AI editor smoke checks"
```

### Task 2: 安装用户级外部工具和 Neovim

**Files:**
- No repository files; install only under `/home/barry/.local` and `/home/barry/.cache`.

- [ ] **Step 1: 安装官方 Neovim 0.12.5 到用户目录**

```bash
set -euo pipefail
download_dir="$(mktemp -d)"
trap 'rm -rf "$download_dir"' EXIT
curl -fL https://github.com/neovim/neovim/releases/download/v0.12.5/nvim-linux-x86_64.tar.gz \
  -o "$download_dir/nvim.tar.gz"
mkdir -p /home/barry/.local/opt
tar -xzf "$download_dir/nvim.tar.gz" -C /home/barry/.local/opt
ln -sfn /home/barry/.local/opt/nvim-linux-x86_64/bin/nvim /home/barry/.local/bin/nvim
hash -r 2>/dev/null || true
```

The existing `/home/barry/.local/bin` is already before `/usr/bin` in the shell
path, so this does not replace the system package.

- [ ] **Step 2: 安装 Ubuntu 工具链**

```bash
sudo apt update
sudo apt install -y clangd clang-format python3-debugpy python3-venv fd-find jq
```

Expose Ubuntu's `fdfind` under the conventional name used by pickers:

```bash
ln -sfn "$(command -v fdfind)" /home/barry/.local/bin/fd
```

- [ ] **Step 3: 安装 lazygit 二进制**

Ubuntu 24.04 的默认仓库不提供当前 lazygit 版本，因此安装最新的官方
Linux x86_64 发布包到用户目录：

```bash
set -euo pipefail
download_dir="$(mktemp -d)"
trap 'rm -rf "$download_dir"' EXIT
lazygit_tag="$(curl -fsSL https://api.github.com/repos/jesseduffield/lazygit/releases/latest | jq -r '.tag_name')"
lazygit_version="${lazygit_tag#v}"
curl -fL "https://github.com/jesseduffield/lazygit/releases/download/${lazygit_tag}/lazygit_${lazygit_version}_Linux_x86_64.tar.gz" \
  -o "$download_dir/lazygit.tar.gz"
tar -xzf "$download_dir/lazygit.tar.gz" -C "$download_dir" lazygit
install -m 0755 "$download_dir/lazygit" /home/barry/.local/bin/lazygit
```

- [ ] **Step 4: 安装 Python 工具和 Codex ACP 适配器**

```bash
uv tool install --upgrade ruff
npm install --global pyright
npm install --global @agentclientprotocol/codex-acp
```

Do not write an API key into any configuration file. Codex authentication stays
managed by its existing login/configuration.

- [ ] **Step 5: 验证每个 executable**

```bash
for tool in nvim codex codex-acp clangd clang-format pyright ruff debugpy gdb cmake rg fd jq; do
  printf '%-14s' "$tool"
  command -v "$tool"
done
nvim --version | sed -n '1,3p'
codex --version
```

Expected: Neovim `0.12.5` or newer, Codex available, and every listed tool has a
resolved path. `debugpy` may be available only as `python3 -m debugpy`; record the
working invocation and use it in DAP configuration.

### Task 3: 修复 Mars.nvim 版本兼容性

**Files:**
- Modify: `nvim/init.lua:14-25,93-125,222-235,304-312`
- Modify: `nvim/lua/mars/options.lua:28-52`
- Modify: `nvim/lua/mars/health.lua:8-31`

- [ ] **Step 1: 让测试继续失败但进入目标版本**

After Task 2, rerun `./tests/smoke.sh`. It must now fail on missing commands rather
than the Neovim version assertion.

- [ ] **Step 2: 固定 Treesitter 到旧 API 兼容分支并加入语言解析器**

Replace the Treesitter spec in `nvim/init.lua` with:

```lua
  {
    'nvim-treesitter/nvim-treesitter',
    branch = 'master',
    build = ':TSUpdate',
    main = 'nvim-treesitter.configs',
    opts = {
      ensure_installed = {
        'bash', 'c', 'cpp', 'cuda', 'diff', 'html', 'lua', 'luadoc',
        'markdown', 'markdown_inline', 'python', 'query', 'vim', 'vimdoc',
      },
      auto_install = true,
      highlight = {
        enable = true,
        additional_vim_regex_highlighting = { 'ruby' },
      },
      indent = { enable = true, disable = { 'ruby' } },
    },
  },
```

The `master` branch remains the compatibility branch for the existing
`nvim-treesitter.configs` setup; the `main` rewrite requires a newer API and is
not used in this plan.

- [ ] **Step 3: 启用文件自动重载和按语言缩进**

Append to `nvim/lua/mars/options.lua`:

```lua
vim.o.autoread = true

vim.api.nvim_create_autocmd({ 'FocusGained', 'BufEnter', 'CursorHold', 'CursorHoldI' }, {
  group = vim.api.nvim_create_augroup('checktime-on-focus', { clear = true }),
  callback = function()
    vim.cmd 'checktime'
  end,
})

vim.api.nvim_create_autocmd('FileType', {
  pattern = 'python',
  callback = function()
    vim.opt_local.tabstop = 4
    vim.opt_local.shiftwidth = 4
    vim.opt_local.softtabstop = 4
    vim.opt_local.expandtab = true
  end,
})

vim.api.nvim_create_autocmd('FileType', {
  pattern = { 'c', 'cpp', 'cuda' },
  callback = function()
    vim.opt_local.tabstop = 2
    vim.opt_local.shiftwidth = 2
    vim.opt_local.softtabstop = 2
    vim.opt_local.expandtab = true
  end,
})
```

- [ ] **Step 4: 更新 health 检查**

Change the version check to require `0.12.0` and add the tools used by this
personal setup:

```lua
local function check_version()
  local version = vim.version()
  if vim.version.ge(version, '0.12.0') then
    vim.health.ok(string.format("Neovim version is: '%s'", tostring(version)))
  else
    vim.health.error(string.format(
      "Neovim 0.12.0 or newer is required; found '%s'",
      tostring(version)
    ))
  end
end

local function check_external_reqs()
  for _, exe in ipairs {
    'git', 'make', 'unzip', 'rg', 'codex', 'clangd', 'clang-format',
    'pyright', 'ruff', 'cmake', 'gdb',
  } do
    if vim.fn.executable(exe) == 1 then
      vim.health.ok(string.format("Found executable: '%s'", exe))
    else
      vim.health.warn(string.format("Could not find executable: '%s'", exe))
    end
  end
end
```

- [ ] **Step 5: 保持插件注册顺序可启动并运行语法检查**

Do not require plugin modules before their files exist. Task 3 should only update
the existing compatibility settings. Task 5 registers
`require 'mars.plugins.codecompanion'` immediately after creating that file, and
Task 6 registers `require 'mars.plugins.git-workbench'` immediately after creating
that file. Run the syntax check after the existing configuration edits:

```bash
NVIM_BIN=/home/barry/.local/bin/nvim ./tests/smoke.sh
```

Expected at this point: the Neovim version assertion passes, while the future
`:CodeCompanionCLI`, `:AI` and `:Git` assertions remain red until Tasks 5-6.

- [ ] **Step 6: Commit compatibility changes**

```bash
git add nvim/init.lua nvim/lua/mars/options.lua nvim/lua/mars/health.lua
git commit -m "fix: align Mars nvim with current runtime"
```

### Task 4: 配置 Python/C++/CUDA LSP、格式化和诊断

**Files:**
- Modify: `nvim/lua/mars/plugins/lsp.lua`
- Modify: `nvim/lua/mars/plugins/lint.lua`
- Modify: `nvim/init.lua:108-124`

- [ ] **Step 1: 移除未使用的 Mason LSP 自动安装依赖**

Keep `mason.nvim` only in the DAP plugin. In the return table at the top of
`nvim/lua/mars/plugins/lsp.lua`, replace the dependency list with:

```lua
  dependencies = {
    {
      'j-hui/fidget.nvim',
      opts = {
        notification = { window = { winblend = 0 } },
      },
    },
    'saghen/blink.cmp',
  },
```

Remove the `mason.nvim`, `mason-lspconfig.nvim` and
`mason-tool-installer.nvim` entries from this plugin. This prevents unrelated
language servers from being downloaded in the personal profile.

- [ ] **Step 2: 写语言服务配置并验证失败状态**

In `nvim/lua/mars/plugins/lsp.lua`, replace the server and Mason auto-install
block with this system-tool configuration:

```lua
    local servers = {
      pyright = {
        settings = {
          python = {
            analysis = {
              autoSearchPaths = true,
              useLibraryCodeForTypes = true,
              diagnosticMode = 'workspace',
            },
          },
        },
      },
      clangd = {
        cmd = {
          'clangd',
          '--background-index',
          '--clang-tidy',
          '--completion-style=detailed',
          '--header-insertion=iwyu',
          '--query-driver=/usr/local/cuda/bin/nvcc,/usr/bin/g++,/usr/bin/gcc',
        },
        filetypes = { 'c', 'cpp', 'objc', 'objcpp', 'cuda' },
        root_markers = { 'compile_commands.json', 'compile_flags.txt', '.git' },
      },
      lua_ls = {
        settings = {
          Lua = {
            completion = { callSnippet = 'Replace' },
          },
        },
      },
    }

    for server_name, server in pairs(servers) do
      server.capabilities = vim.tbl_deep_extend('force', {}, capabilities, server.capabilities or {})
      vim.lsp.config(server_name, server)
      vim.lsp.enable(server_name)
    end
```

Remove `gopls`, `ts_ls`, `eslint` and `ruff` from this personal profile so missing
unrelated servers cannot trigger background installation. Keep Mason as an
optional dependency for DAP packages used later.

- [ ] **Step 3: 配置 Ruff 诊断**

Set `lint.linters_by_ft` in `nvim/lua/mars/plugins/lint.lua` to:

```lua
      lint.linters_by_ft = {
        python = { 'ruff' },
        markdown = { 'markdownlint' },
      }
```

Retain the existing `BufEnter`, `BufWritePost` and `InsertLeave` trigger group.

- [ ] **Step 4: 启用 Conform 格式化**

Replace the existing format-on-save and formatter map in `nvim/init.lua` with:

```lua
      format_on_save = function(bufnr)
        if vim.b[bufnr].disable_autoformat then
          return
        end
        return { timeout_ms = 1000, lsp_format = 'fallback' }
      end,
      formatters_by_ft = {
        lua = { 'stylua' },
        python = { 'ruff_organize_imports', 'ruff_format' },
        c = { 'clang_format' },
        cpp = { 'clang_format' },
        cuda = { 'clang_format' },
        go = { 'goimports', 'goimports-reviser', 'gofumpt' },
        javascript = { 'prettierd', 'prettier', stop_after_first = true },
        typescript = { 'prettierd', 'prettier', stop_after_first = true },
      },
```

The `<leader>f` mapping remains the explicit manual format action.

- [ ] **Step 5: 安装并验证 parser/LSP/formatter 状态**

```bash
NVIM_BIN=/home/barry/.local/bin/nvim ./tests/smoke.sh
```

Then manually open one file of each type and check:

```vim
:checkhealth vim.lsp
:LspInfo
:ConformInfo
:set filetype?
```

Expected: `pyright` attaches to Python, `clangd` attaches to C++/CUDA when a
root marker or `compile_commands.json` is present, and formatters resolve to the
installed executables.

- [ ] **Step 6: Commit language support**

```bash
git add nvim/init.lua nvim/lua/mars/plugins/lsp.lua nvim/lua/mars/plugins/lint.lua
git commit -m "feat: add Python C++ and CUDA editor tooling"
```

### Task 5: 建立常驻 Codex 面板和个人 AI 命令

**Files:**
- Create: `nvim/lua/mars/plugins/codecompanion.lua`
- Create: `nvim/lua/mars/ai.lua`
- Modify: `nvim/lua/mars/plugins/ai.lua`
- Modify: `nvim/init.lua:147-156`

- [ ] **Step 1: 先注册 CodeCompanion CLI agent**

Create `nvim/lua/mars/plugins/codecompanion.lua`:

```lua
local function codex_command()
  local path = vim.fn.exepath('codex')
  return path ~= '' and path or 'codex'
end

return {
  {
    'olimorris/codecompanion.nvim',
    version = '^19.0.0',
    event = 'VimEnter',
    dependencies = {
      'nvim-lua/plenary.nvim',
      'nvim-treesitter/nvim-treesitter',
    },
    opts = {
      interactions = {
        cli = {
          agent = 'codex',
          agents = {
            codex = {
              cmd = codex_command(),
              args = { '--no-alt-screen' },
              description = 'OpenAI Codex CLI',
              provider = 'terminal',
            },
          },
          opts = {
            auto_insert = true,
            reload = true,
          },
        },
      },
      display = {
        cli = {
          window = {
            layout = 'vertical',
            width = 0.40,
            full_height = true,
            opts = {
              number = false,
              relativenumber = false,
              signcolumn = 'no',
              wrap = true,
            },
          },
        },
      },
      opts = { log_level = 'WARN' },
    },
    config = function(_, opts)
      require('codecompanion').setup(opts)

      local ai = require('mars.ai')
      vim.keymap.set('n', '<leader>cc', ai.toggle, { desc = 'Codex: toggle panel' })
      vim.keymap.set('n', '<leader>cf', ai.focus, { desc = 'Codex: focus panel' })
      vim.keymap.set('n', '<leader>ca', ai.ask, { desc = 'Codex: ask with context' })
      vim.keymap.set({ 'n', 'v' }, '<leader>cp', ai.prompt, { desc = 'Codex: prompt' })
      vim.keymap.set('n', '<leader>cr', ai.review, { desc = 'Codex: review diff' })
      vim.keymap.set('n', '<leader>cm', ai.search_markdown, { desc = 'Codex: search Markdown' })
    end,
  },
}
```

`--no-alt-screen` keeps terminal scrollback usable inside Neovim. The panel's
working directory is the Neovim working directory; daily usage starts Neovim at
the project root (`nvim .`).

- [ ] **Step 2: 实现 AI helper，不调用私有 CodeCompanion API**

Create `nvim/lua/mars/ai.lua`:

```lua
local M = {}

local function root()
  return vim.fs.root(0, { '.git', 'pyproject.toml', 'CMakeLists.txt', 'Makefile' })
    or vim.fn.getcwd()
end

local function send(prompt, opts)
  opts = vim.tbl_extend('force', { agent = 'codex', submit = true }, opts or {})
  require('codecompanion').cli(prompt, opts)
end

function M.toggle()
  require('codecompanion').toggle_cli({ agent = 'codex' })
end

function M.focus()
  for _, win in ipairs(vim.api.nvim_list_wins()) do
    local bufnr = vim.api.nvim_win_get_buf(win)
    if vim.bo[bufnr].filetype == 'codecompanion_cli' then
      vim.api.nvim_set_current_win(win)
      vim.cmd 'startinsert'
      return
    end
  end
  M.toggle()
end

function M.ask()
  require('codecompanion').cli({ agent = 'codex', prompt = true })
end

function M.prompt()
  require('codecompanion').cli({ agent = 'codex', prompt = true })
end

function M.review()
  send('#{diff} Review the current Git diff. List blocking defects first, then tests and a commit summary.')
end

function M.search_markdown()
  vim.ui.input({ prompt = 'Markdown search: ' }, function(query)
    if not query or vim.trim(query) == '' then
      return
    end

    vim.system({
      'rg', '--color=never', '--line-number', '--column', '--no-heading',
      '--glob', '*.md', '--glob', '!.git/**', '--glob', '!**/.env*',
      '--glob', '!**/node_modules/**', '--max-count', '200', query, root(),
    }, { text = true }, function(result)
      vim.schedule(function()
        if result.code ~= 0 and result.code ~= 1 then
          vim.notify('Markdown search failed: ' .. vim.trim(result.stderr or ''), vim.log.levels.ERROR)
          return
        end

        local output = vim.trim(result.stdout or '')
        if output == '' then
          vim.notify('No Markdown matches for: ' .. query, vim.log.levels.INFO)
          return
        end

        send(string.format(
          'Search results for `%s` from Markdown files under `%s`:\n\n```text\n%s\n```\n\n'
            .. 'Use these paths and line numbers as evidence, then answer the request.',
          query,
          root(),
          output
        ))
      end)
    end)
  end)
end

function M.dispatch(opts)
  local subcommand = opts.fargs[1]
  local rest = table.concat(vim.list_slice(opts.fargs, 2), ' ')
  if subcommand == 'search-md' then
    return M.search_markdown()
  elseif subcommand == 'review' then
    return M.review()
  elseif subcommand == 'ask' or subcommand == 'plan' or subcommand == 'edit'
    or subcommand == 'test' or subcommand == 'git' then
    local prefix = ({
      ask = 'Answer using the current project context: ',
      plan = 'Complete the requirements, state assumptions, and produce an implementation plan: ',
      edit = 'Propose an edit and show the intended diff before applying it: ',
      test = 'Run or explain the appropriate tests and return actionable failures: ',
      git = 'Explain the Git state and prepare the next safe Git action: ',
    })[subcommand]
    return send('#{buffer} #{diagnostics} ' .. prefix .. rest)
  end
  if subcommand then
    return send(table.concat(opts.fargs, ' '))
  end
  return M.ask()
end

vim.api.nvim_create_user_command('AI', M.dispatch, {
  nargs = '*',
  complete = function()
    return { 'ask', 'edit', 'git', 'plan', 'review', 'search-md', 'test' }
  end,
  desc = 'Open or send a prompt to the Codex panel',
})

return M
```

The helper uses vectorized `vim.system` arguments, caps search output at 200
matches, and excludes common secret/build directories. It uses only the public
`require('codecompanion').cli` and `toggle_cli` APIs.

- [ ] **Step 3: 停用旧 AI 快捷键冲突**

In `nvim/lua/mars/plugins/ai.lua`, set the Claude Code and Gemini specs to
`enabled = false` for the initial profile. In the `blink.cmp` dependency block in
`nvim/init.lua`, set the nested `supermaven-inc/supermaven-nvim` spec to
`enabled = false`. Preserve all configuration blocks so they can be re-enabled
without reconstructing them. Codex owns the `<leader>c` namespace.

- [ ] **Step 4: 运行插件同步和命令测试**

```bash
NVIM_BIN=/home/barry/.local/bin/nvim ./tests/smoke.sh
```

Then manually verify inside Neovim:

```vim
:CodeCompanionCLI
:AI plan 为当前项目补全需求和测试标准
:AI search-md CUDA 内存
```

Expected: `<leader>cc` hides/shows the same Codex process; hiding the split does
not terminate it; `:AI search-md` sends bounded Markdown evidence to that session.

- [ ] **Step 5: Commit the Codex panel**

```bash
git add nvim/lua/mars/plugins/codecompanion.lua nvim/lua/mars/ai.lua nvim/lua/mars/plugins/ai.lua
git commit -m "feat: add persistent Codex CLI panel"
```

### Task 6: 完成 Git、搜索和任务入口

**Files:**
- Create: `nvim/lua/mars/plugins/git-workbench.lua`
- Modify: `nvim/init.lua`
- Modify: `README.md`
- Modify: `README_zh.md`

- [ ] **Step 1: 添加 Fugitive 和 Snacks Git 入口**

Create `nvim/lua/mars/plugins/git-workbench.lua`:

```lua
return {
  {
    'tpope/vim-fugitive',
    cmd = { 'Git', 'G' },
  },
  {
    'folke/snacks.nvim',
    priority = 1000,
    lazy = false,
    opts = {
      input = { enabled = true },
      picker = { enabled = true },
      terminal = { enabled = true },
      lazygit = { enabled = true },
    },
    keys = {
      { '<leader>gg', function() Snacks.lazygit() end, desc = 'Git: open lazygit' },
      { '<leader>gf', function() Snacks.picker.git_files() end, desc = 'Git: find tracked files' },
      { '<leader>gb', function() Snacks.picker.git_branches() end, desc = 'Git: switch branch' },
      { '<leader>gd', function() Snacks.picker.git_diff() end, desc = 'Git: inspect diff' },
    },
  },
}
```

Retain existing gitsigns and Neogit mappings. If `<leader>gg` currently belongs to
another mapping, move that mapping to `<leader>gG` so lazygit has one predictable
entry.

- [ ] **Step 2: 统一搜索入口**

Add these mappings to `nvim/init.lua` after Snacks is loaded:

```lua
vim.keymap.set('n', '<leader>ff', function() Snacks.picker.files({ cwd = vim.fn.getcwd() }) end,
  { desc = 'Files: find' })
vim.keymap.set('n', '<leader>fg', function() Snacks.picker.grep({ cwd = vim.fn.getcwd() }) end,
  { desc = 'Files: grep' })
vim.keymap.set('n', '<leader>fm', function() Snacks.picker.grep({ cwd = vim.fn.getcwd(), glob = '*.md' }) end,
  { desc = 'Files: grep Markdown' })
```

Do not remove Telescope immediately; existing LSP references and user mappings stay
available until the Snacks equivalents have been manually verified.

- [ ] **Step 3: 配置 shell completion**

Verify that the existing Oh My Zsh `git` plugin exposes `_git`, then generate the
Codex completion file without adding credentials:

```bash
completion_dir="${ZDOTDIR:-/home/barry}/.zsh/completions"
mkdir -p "$completion_dir"
codex completion zsh > "$completion_dir/_codex"
```

Add the completion directory to `fpath` in `~/.zshrc` only when it is absent:

```zsh
fpath=("${ZDOTDIR:-$HOME}/.zsh/completions" $fpath)
autoload -Uz compinit && compinit
```

Reload with `exec zsh`, then verify `git <Tab>` and `codex <Tab>` show subcommands
and file/branch arguments.

- [ ] **Step 4: 评估任务插件但不强制加入**

Use existing CMake/Make/pytest commands first. Add `stevearc/overseer.nvim` only if
the following manual workflow is needed after the basic panel works:

```vim
:OverseerRun
:OverseerToggle
```

Do not add a task plugin merely to duplicate terminal commands in the MVP.

- [ ] **Step 5: Commit Git and search integration**

```bash
git add nvim/init.lua nvim/lua/mars/plugins/git-workbench.lua README.md README_zh.md
git commit -m "feat: add keyboard Git and project search workflow"
```

### Task 7: 安全策略、会话恢复和 ACP 评估

**Files:**
- Modify: `tests/smoke.sh`
- Modify: `README.md`, `README_zh.md`

- [ ] **Step 1: 写安全回归测试**

Extend `tests/smoke.sh` with a source check that fails if tracked Lua files contain
common credential assignments:

```bash
if rg -n --glob '*.lua' '(OPENAI_API_KEY|CODEX_API_KEY|api_key|token)[[:space:]]*=' "$repo_root/nvim"; then
  echo 'credential-like assignment found in Neovim config' >&2
  exit 1
fi
```

- [ ] **Step 2: 保持 Codex CLI 权限由 Codex 管理**

Do not pass `--dangerously-bypass-approvals-and-sandbox` from Neovim. Keep normal
Codex approval prompts and require confirmation for `commit`, `push`, hard reset,
deletion and writes outside the current project.

- [ ] **Step 3: 评估 ACP 而不改变日常后端**

Verify the separately installed adapter and its protocol without changing the
daily CLI spec:

```bash
codex-acp --version
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{},"clientInfo":{"name":"mars-smoke","version":"0.1"}}}' \
  | timeout 10s codex-acp
```

Record whether initialization succeeds, then test session creation, permission
requests, file reads/writes, interruption and command output in a temporary
project. Do not promote ACP to the default until every event has a visible and
recoverable Neovim representation.

- [ ] **Step 4: Commit safety and ACP evaluation notes**

```bash
git add tests/smoke.sh README.md README_zh.md
git commit -m "test: enforce safe Codex workflow"
```

### Task 8: 端到端验证、合并和交付记录

**Files:**
- Modify: `README.md`, `README_zh.md`
- Modify: `nvim/lua/mars/health.lua` if verification discovers a misleading check.

- [ ] **Step 1: 同步锁定插件并运行静态检查**

From the worktree:

```bash
NVIM_BIN=/home/barry/.local/bin/nvim ./tests/smoke.sh
./tests/context.sh
git diff --check
```

Run the plugin manager once interactively with `:Lazy sync`, then record the
generated `nvim/lazy-lock.json` locally. Because the Mars repository ignores this
file, do not force-add it unless the user explicitly wants reproducible plugin
commits.

- [ ] **Step 2: 验证全键盘验收场景**

In a real Git project, execute this sequence and record results:

```text
nvim .
<leader>cc                  open Codex panel
:AI plan <natural-language goal>
<leader>cm                  search Markdown and send evidence
<leader>cp                  prompt with current file/selection
<leader>cr                  review current diff
<leader>gg                  inspect Git state in lazygit
<leader>f                   format Python/C++/CUDA buffer
<F5>                        run configured DAP session when available
```

Verify that hiding the panel preserves the process, external Codex edits reload
buffers, diagnostics and test output can be sent back with `#{diagnostics}` and
`#{terminal}`, and dangerous Git actions still prompt.

- [ ] **Step 3: 验证项目类型**

Open all of the following and check filetype/LSP/root behavior:

```text
*.py                  pyright + ruff + ruff_format
*.cpp / *.h           clangd + clang-format
*.cu                  clangd with CUDA query-driver and compile_commands.json
README.md / docs/*.md rg/Snacks Markdown search
```

For CUDA projects, verify the nearest `compile_commands.json` contains the actual
`nvcc` include paths, macros and architecture flags before trusting diagnostics.

- [ ] **Step 4: 合并隔离分支到实际配置**

Only after all previous checks pass:

```bash
git -C /home/barry/.config/mars.nvim merge --ff-only feature/mars-ai-editor
git -C /home/barry/.config/mars.nvim status --short --branch
```

The symlink `~/.config/nvim -> /home/barry/.config/mars.nvim/nvim` then points at
the verified configuration without copying files or touching unrelated project
changes.

- [ ] **Step 5: 记录交付状态**

Update both Mars README files with the actual installed versions, working keymaps,
known limitations, and the commands used for `:checkhealth`, LSP, Codex, Git and
Markdown search. Commit the final documentation:

```bash
git -C /home/barry/.config/mars.nvim add README.md README_zh.md nvim
git -C /home/barry/.config/mars.nvim commit -m "docs: record AI CLI editor setup"
```

## 完成判据

The plan is complete only when all of these have direct command/runtime evidence:

- Neovim reports 0.12.x or newer and Mars.nvim starts without Lua errors.
- `:CodeCompanionCLI`, `:AI`, `:Git` and the documented keymaps work.
- Codex runs in one persistent Neovim-managed terminal and can be hidden/restored.
- Current file, selection, diagnostics, terminal output, Git diff and Markdown
  search results can be sent to the same Codex session.
- Python and C++/CUDA files receive LSP completion/diagnostics and configured
  formatting; CMake compile databases are respected.
- Git completion works in the shell; Neovim exposes hunk, diff, status and lazygit
  workflows without mouse input.
- No credentials appear in tracked config or documentation.
- `tests/smoke.sh`, `tests/context.sh` and `git diff --check` pass after merge.
- Existing unrelated files in the parent project remain untouched.
