"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
    createController,
    getCodeText,
} = require("./codex-keyboard-tools.js");

function createScrollContainer(options = {}) {
    const { hidden = false } = options;
    return {
        clientHeight: 500,
        hidden,
        scrollHeight: 2000,
        scrollTop: -400,
        style: {
            getPropertyValue() {
                return "0px";
            },
        },
        parentElement: null,
        contains(node) {
            for (let current = node; current; current = current.parentElement) {
                if (current === this) return true;
            }
            return false;
        },
        closest() {
            return this.hidden ? this : null;
        },
        getBoundingClientRect() {
            return this.hidden
                ? { top: 0, bottom: 0, height: 0, width: 0 }
                : { top: 0, bottom: 500, height: 500, width: 400 };
        },
        scrollBy({ top }) {
            this.scrollTop += top;
        },
        scrollTo({ top }) {
            this.scrollTop = top;
        },
    };
}

function createCodeBlock(text, top, parentElement, options = {}) {
    const {
        codeIndex = null,
        hidden = false,
        scopeElement = null,
        turnKey = null,
    } = options;
    const attributes = new Map([
        ["data-markdown-copy", "code-block"],
        ["data-markdown-copy-text", text],
    ]);
    const turnElement = scopeElement ?? (turnKey == null
        ? null
        : {
            parentElement,
            getAttribute(name) {
                return name === "data-turn-key" ? turnKey : null;
            },
        });
    const codeElement = {
        textContent: `fallback:${text}`,
        getAttribute(name) {
            return name === "data-code-block-index" && codeIndex != null
                ? String(codeIndex)
                : null;
        },
    };

    return {
        hidden,
        isConnected: true,
        parentElement: turnElement ?? parentElement,
        textContent: `decorated:${text}`,
        focused: false,
        scrolledIntoView: false,
        getAttribute(name) {
            return attributes.get(name) ?? null;
        },
        setAttribute(name, value) {
            attributes.set(name, String(value));
        },
        removeAttribute(name) {
            attributes.delete(name);
        },
        closest(selector) {
            if (this.hidden) return this;
            return turnElement && (
                selector === "[data-turn-key]"
                || selector === "[data-content-search-turn-key]"
                || selector === "[data-chatgpt-conversation-turn-id]"
            ) ? turnElement : null;
        },
        querySelector(selector) {
            return selector === "code" || selector === "[data-code-block-index]"
                ? codeElement
                : null;
        },
        top,
        getBoundingClientRect() {
            return { top: this.top, bottom: this.top + 100, height: 100, width: 300 };
        },
        scrollIntoView() {
            this.scrolledIntoView = true;
        },
        focus() {
            this.focused = true;
        },
    };
}

function createHarness() {
    const scrollContainer = createScrollContainer();
    const blocks = [
        createCodeBlock("alpha\n", 50, scrollContainer),
        createCodeBlock("beta\n", 350, scrollContainer),
    ];
    const clipboardWrites = [];
    const document = {
        body: {},
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return blocks;
            if (selector === "[data-pip-anchor-host]") return [scrollContainer];
            if (selector === "div, main, section") return [scrollContainer];
            return [];
        },
    };
    const window = {
        document,
        innerHeight: 1000,
        navigator: {
            clipboard: {
                async writeText(text) {
                    clipboardWrites.push(text);
                },
            },
        },
    };
    const controller = createController(window, { showStatus: false });

    return { blocks, clipboardWrites, controller, scrollContainer };
}

function createKeyEvent(key, overrides = {}) {
    return {
        key,
        altKey: true,
        ctrlKey: false,
        metaKey: false,
        shiftKey: false,
        repeat: false,
        defaultPrevented: false,
        prevented: false,
        stopped: false,
        preventDefault() {
            this.prevented = true;
        },
        stopImmediatePropagation() {
            this.stopped = true;
        },
        ...overrides,
    };
}

test("extracts only the raw code block text", () => {
    const scrollContainer = createScrollContainer();
    const block = createCodeBlock("printf(\"ok\");\n", 0, scrollContainer);

    assert.equal(getCodeText(block), "printf(\"ok\");\n");
});

test("selects the viewport-nearest code block then cycles", () => {
    const { blocks, controller } = createHarness();

    assert.equal(controller.selectCodeBlock(1), blocks[1]);
    assert.equal(controller.selectCodeBlock(1), blocks[0]);
    assert.equal(controller.selectCodeBlock(-1), blocks[1]);
    assert.equal(blocks[1].getAttribute("data-codex-keyboard-active-code"), "true");
    assert.equal(blocks[1].focused, true);
    assert.equal(blocks[1].scrolledIntoView, false);
});

test("continues cycling after the active code block DOM node is remounted", () => {
    const scrollContainer = createScrollContainer();
    let blocks = [
        createCodeBlock("alpha\n", 50, scrollContainer),
        createCodeBlock("beta\n", 350, scrollContainer),
    ];
    const document = {
        body: {},
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return blocks;
            if (selector === "[data-pip-anchor-host]") return [scrollContainer];
            if (selector === "div, main, section") return [scrollContainer];
            return [];
        },
    };
    const controller = createController({ document, innerHeight: 1000 }, { showStatus: false });

    assert.equal(controller.selectCodeBlock(1), blocks[1]);
    blocks[1] = createCodeBlock("beta\n", 350, scrollContainer);

    assert.equal(controller.selectCodeBlock(1), blocks[0]);
});

test("tracks the logical code block when virtualization inserts a block before it", () => {
    const scrollContainer = createScrollContainer();
    const makeTurn = (key) => ({
        blocks: [],
        parentElement: scrollContainer,
        getAttribute(name) {
            return name === "data-turn-key" ? key : null;
        },
        querySelectorAll(selector) {
            return selector === '[data-markdown-copy="code-block"]' ? this.blocks : [];
        },
    });
    const makeBlock = (text, top, turn) => {
        const block = createCodeBlock(text, top, scrollContainer, { scopeElement: turn });
        turn.blocks.push(block);
        return block;
    };
    const turnA = makeTurn("turn-a");
    const turnB = makeTurn("turn-b");
    let blocks = [
        makeBlock("alpha\n", 50, turnA),
        makeBlock("beta\n", 350, turnB),
    ];
    const document = {
        body: {},
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return blocks;
            if (selector === "[data-pip-anchor-host]") return [scrollContainer];
            if (selector === "div, main, section") return [scrollContainer];
            return [];
        },
    };
    const controller = createController({ document, innerHeight: 1000 }, { showStatus: false });

    assert.equal(controller.selectCodeBlock(1), blocks[1]);
    const turnX = makeTurn("turn-x");
    const nextTurnA = makeTurn("turn-a");
    const nextTurnB = makeTurn("turn-b");
    blocks = [
        makeBlock("inserted\n", 30, turnX),
        makeBlock("alpha\n", 150, nextTurnA),
        makeBlock("beta\n", 350, nextTurnB),
    ];

    assert.equal(controller.selectCodeBlock(1), blocks[0]);
});

test("retries reveal against a remounted block during the layout correction frame", () => {
    const scrollContainer = createScrollContainer();
    let blocks = [
        createCodeBlock("target\n", 100, scrollContainer, { codeIndex: 0, turnKey: "turn-a" }),
    ];
    const animationFrames = [];
    const document = {
        body: {},
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return blocks;
            if (selector === "[data-pip-anchor-host]") return [scrollContainer];
            if (selector === "div, main, section") return [scrollContainer];
            return [];
        },
    };
    const controller = createController({
        document,
        innerHeight: 1000,
        requestAnimationFrame(callback) {
            animationFrames.push(callback);
            return animationFrames.length;
        },
    }, { showStatus: false });

    controller.selectCodeBlock(1);
    assert.equal(scrollContainer.scrollTop, -400);

    const oldBlock = blocks[0];
    oldBlock.isConnected = false;
    blocks[0] = createCodeBlock("target\n", 520, scrollContainer, {
        codeIndex: 0,
        turnKey: "turn-a",
    });
    animationFrames.shift()();

    assert.equal(scrollContainer.scrollTop, -256);
});

test("preserves the logical selection across a transient empty code-block list", () => {
    const scrollContainer = createScrollContainer();
    let blocks = [
        createCodeBlock("alpha\n", 50, scrollContainer, { codeIndex: 0, turnKey: "turn-a" }),
        createCodeBlock("beta\n", 350, scrollContainer, { codeIndex: 0, turnKey: "turn-b" }),
    ];
    const document = {
        body: {},
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return blocks;
            if (selector === "[data-pip-anchor-host]") return [scrollContainer];
            if (selector === "div, main, section") return [scrollContainer];
            return [];
        },
    };
    const controller = createController({ document, innerHeight: 1000 }, { showStatus: false });

    assert.equal(controller.selectCodeBlock(1), blocks[1]);
    blocks = [];
    assert.equal(controller.selectCodeBlock(1), null);
    blocks = [
        createCodeBlock("alpha\n", 50, scrollContainer, { codeIndex: 0, turnKey: "turn-a" }),
        createCodeBlock("beta\n", 350, scrollContainer, { codeIndex: 0, turnKey: "turn-b" }),
    ];

    assert.equal(controller.selectCodeBlock(1), blocks[0]);
});

test("reveals a selected block with minimal timeline scrolling and one layout correction", () => {
    const scrollContainer = createScrollContainer();
    const block = createCodeBlock("target\n", 520, scrollContainer);
    const animationFrames = [];
    scrollContainer.scrollBy = ({ top }) => {
        scrollContainer.scrollTop += top;
        block.top -= top;
    };
    const document = {
        body: {},
        activeElement: block,
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return [block];
            if (selector === "[data-pip-anchor-host]") return [scrollContainer];
            if (selector === "div, main, section") return [scrollContainer];
            return [];
        },
    };
    const controller = createController({
        document,
        innerHeight: 1000,
        requestAnimationFrame(callback) {
            animationFrames.push(callback);
            return animationFrames.length;
        },
    }, { showStatus: false });

    controller.selectCodeBlock(1);
    assert.equal(scrollContainer.scrollTop, -256);
    assert.equal(block.top, 376);
    assert.equal(block.scrolledIntoView, false);

    block.top += 60;
    animationFrames.shift()();
    assert.equal(scrollContainer.scrollTop, -196);
    assert.equal(block.top, 376);
});

test("does not move the timeline when the selected block is already visible", () => {
    const { blocks, controller, scrollContainer } = createHarness();

    controller.selectCodeBlock(1);
    assert.equal(blocks[1].getAttribute("data-codex-keyboard-active-code"), "true");
    assert.equal(scrollContainer.scrollTop, -400);
});

test("reveals a selected block through its scrollable ancestor without an anchor host", () => {
    const scrollContainer = createScrollContainer();
    const block = createCodeBlock("target\n", 520, scrollContainer);
    const document = {
        body: {},
        activeElement: null,
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return [block];
            if (selector === "[data-pip-anchor-host]") return [];
            if (selector === "div, main, section") return [scrollContainer];
            return [];
        },
    };
    const controller = createController({
        document,
        innerHeight: 1000,
    }, { showStatus: false });

    controller.selectCodeBlock(1);
    assert.equal(scrollContainer.scrollTop, -256);
});

test("falls back from a non-scrollable anchor host to its scrollable ancestor", () => {
    const scrollContainer = createScrollContainer();
    const anchorHost = createScrollContainer();
    anchorHost.scrollHeight = anchorHost.clientHeight;
    anchorHost.parentElement = scrollContainer;
    const block = createCodeBlock("target\n", 520, anchorHost);
    const document = {
        body: {},
        activeElement: block,
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return [block];
            if (selector === "[data-pip-anchor-host]") return [anchorHost];
            if (selector === "div, main, section") return [anchorHost, scrollContainer];
            return [];
        },
    };
    const controller = createController({ document, innerHeight: 1000 }, { showStatus: false });

    controller.selectCodeBlock(1);
    assert.equal(scrollContainer.scrollTop, -256);
    assert.equal(anchorHost.scrollTop, -400);
});

test("copies only the active single code block", async () => {
    const { clipboardWrites, controller } = createHarness();

    controller.selectCodeBlock(1);
    assert.equal(await controller.copyActiveCode(), true);
    assert.deepEqual(clipboardWrites, ["beta\n"]);
});

test("scrolls by lines and pages and jumps to the latest message", () => {
    const { controller, scrollContainer } = createHarness();

    assert.equal(controller.scrollByPixels(160), true);
    assert.equal(scrollContainer.scrollTop, -240);
    assert.equal(controller.scrollByPixels(-160), true);
    assert.equal(scrollContainer.scrollTop, -400);
    assert.equal(controller.scrollByPage(1), true);
    assert.equal(scrollContainer.scrollTop, 0);
    assert.equal(controller.scrollByPage(-1), true);
    assert.equal(scrollContainer.scrollTop, -400);
    assert.equal(controller.jumpLatest(), true);
    assert.equal(scrollContainer.scrollTop, 0);
});

test("handles Alt+Home and Alt+End without consuming plain boundary keys", () => {
    const { controller, scrollContainer } = createHarness();
    const homeEvent = createKeyEvent("Home");
    const endEvent = createKeyEvent("End");

    assert.equal(controller.handleKeydown(homeEvent), true);
    assert.equal(scrollContainer.scrollTop, -1500);
    assert.equal(homeEvent.prevented, true);

    assert.equal(controller.handleKeydown(endEvent), true);
    assert.equal(scrollContainer.scrollTop, 0);
    assert.equal(endEvent.prevented, true);

    const plainHome = createKeyEvent("Home", { altKey: false });
    const plainEnd = createKeyEvent("End", { altKey: false });
    assert.equal(controller.handleKeydown(plainHome), false);
    assert.equal(controller.handleKeydown(plainEnd), false);
});

test("scrolls exactly 80 percent of a narrow Webview page", () => {
    const { controller, scrollContainer } = createHarness();
    scrollContainer.clientHeight = 200;

    assert.equal(controller.scrollByPage(1), true);
    assert.equal(scrollContainer.scrollTop, -240);
});

test("excludes hidden code blocks outside the visible timeline", async () => {
    const visibleTimeline = createScrollContainer();
    const hiddenTimeline = createScrollContainer({ hidden: true });
    const visibleBlock = createCodeBlock("visible\n", 50, visibleTimeline);
    const hiddenBlock = createCodeBlock("hidden\n", 450, hiddenTimeline, { hidden: true });
    const clipboardWrites = [];
    const document = {
        body: {},
        activeElement: visibleBlock,
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') {
                return [visibleBlock, hiddenBlock];
            }
            if (selector === "[data-pip-anchor-host]") {
                return [visibleTimeline, hiddenTimeline];
            }
            if (selector === "div, main, section") {
                return [visibleTimeline, hiddenTimeline];
            }
            return [];
        },
    };
    const controller = createController({
        document,
        innerHeight: 1000,
        navigator: {
            clipboard: {
                async writeText(text) {
                    clipboardWrites.push(text);
                },
            },
        },
    }, { showStatus: false });

    assert.equal(await controller.copyActiveCode(), true);
    assert.deepEqual(clipboardWrites, ["visible\n"]);
});

test("scrolling ignores a stale block from a newly hidden timeline", () => {
    const oldTimeline = createScrollContainer();
    const currentTimeline = createScrollContainer();
    const oldBlock = createCodeBlock("old\n", 450, oldTimeline);
    const currentBlock = createCodeBlock("current\n", 50, currentTimeline);
    const document = {
        body: {},
        activeElement: oldBlock,
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return [oldBlock, currentBlock];
            if (selector === "[data-pip-anchor-host]") return [oldTimeline, currentTimeline];
            if (selector === "div, main, section") return [oldTimeline, currentTimeline];
            return [];
        },
    };
    const controller = createController({ document, innerHeight: 1000 }, { showStatus: false });
    controller.selectCodeBlock(1);
    const oldTimelinePosition = oldTimeline.scrollTop;

    oldTimeline.hidden = true;
    oldBlock.hidden = true;
    document.activeElement = currentBlock;
    assert.equal(controller.scrollByPixels(160), true);
    assert.equal(oldTimeline.scrollTop, oldTimelinePosition);
    assert.equal(currentTimeline.scrollTop, -240);
});

test("short visible timelines still scope single-code-block copying", async () => {
    const shortTimeline = createScrollContainer();
    shortTimeline.scrollHeight = shortTimeline.clientHeight;
    const previewContainer = createScrollContainer();
    const threadBlock = createCodeBlock("thread\n", 50, shortTimeline);
    const externalPreviewBlock = createCodeBlock("preview\n", 450, previewContainer);
    const clipboardWrites = [];
    const document = {
        body: {},
        activeElement: threadBlock,
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') {
                return [threadBlock, externalPreviewBlock];
            }
            if (selector === "[data-pip-anchor-host]") return [shortTimeline];
            if (selector === "div, main, section") return [shortTimeline, previewContainer];
            return [];
        },
    };
    const controller = createController({
        document,
        innerHeight: 1000,
        navigator: {
            clipboard: {
                async writeText(text) {
                    clipboardWrites.push(text);
                },
            },
        },
    }, { showStatus: false });

    assert.equal(await controller.copyActiveCode(), true);
    assert.deepEqual(clipboardWrites, ["thread\n"]);
});

test("copies an empty fenced code block exactly", async () => {
    const scrollContainer = createScrollContainer();
    const emptyBlock = createCodeBlock("", 450, scrollContainer);
    const clipboardWrites = [];
    const document = {
        body: {},
        activeElement: emptyBlock,
        querySelectorAll(selector) {
            if (selector === '[data-markdown-copy="code-block"]') return [emptyBlock];
            if (selector === "[data-pip-anchor-host]") return [scrollContainer];
            if (selector === "div, main, section") return [scrollContainer];
            return [];
        },
    };
    const controller = createController({
        document,
        innerHeight: 1000,
        navigator: {
            clipboard: {
                async writeText(text) {
                    clipboardWrites.push(text);
                },
            },
        },
    }, { showStatus: false });

    assert.equal(await controller.copyActiveCode(), true);
    assert.deepEqual(clipboardWrites, [""]);
});

test("handles Alt shortcuts and ignores plain keys", async () => {
    const { clipboardWrites, controller, scrollContainer } = createHarness();
    const copyEvent = createKeyEvent("c");

    assert.equal(controller.handleKeydown(copyEvent), true);
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(clipboardWrites, ["beta\n"]);
    assert.equal(copyEvent.prevented, true);
    assert.equal(copyEvent.stopped, false);

    const scrollEvent = createKeyEvent("j");
    assert.equal(controller.handleKeydown(scrollEvent), true);
    assert.equal(scrollContainer.scrollTop, -240);

    const plainEvent = createKeyEvent("j", { altKey: false });
    assert.equal(controller.handleKeydown(plainEvent), false);
    assert.equal(plainEvent.prevented, false);
});

test("handles Alt code-block shortcuts even when default was already prevented", () => {
    const { blocks, controller } = createHarness();
    const event = createKeyEvent("n", { defaultPrevented: true });

    assert.equal(controller.handleKeydown(event), true);
    assert.equal(controller.getActiveBlock(), blocks[1]);
    assert.equal(event.prevented, true);
    assert.equal(event.stopped, false);
});

test("dispatches session navigation through the local Webview message bus", () => {
    const messages = [];
    const document = {
        body: {},
        querySelectorAll() {
            return [];
        },
    };
    const window = {
        document,
        location: { origin: "vscode-webview://test" },
        postMessage(message, targetOrigin) {
            messages.push({ message, targetOrigin });
        },
    };
    const sessionController = createController(window, { showStatus: false });
    const searchEvent = createKeyEvent("s", { ctrlKey: true });
    const backEvent = createKeyEvent("[", { altKey: false, ctrlKey: true });

    assert.equal(sessionController.handleKeydown(searchEvent), true);
    assert.equal(sessionController.handleKeydown(backEvent), true);
    assert.deepEqual(messages, [
        {
            message: { type: "chat-search-command-menu" },
            targetOrigin: "vscode-webview://test",
        },
        {
            message: { type: "navigate-back" },
            targetOrigin: "vscode-webview://test",
        },
    ]);
    assert.equal(searchEvent.prevented, true);
    assert.equal(backEvent.prevented, true);
});

test("injects the Webview module exactly once before the main module", () => {
    const {
        INJECTED_SCRIPT_TAG,
        patchIndexHtml,
    } = require("./patch-codex-webview-keyboard-tools");
    const original = [
        "<html>",
        "  <head>",
        '    <script type="module" crossorigin src="./assets/index-example.js"></script>',
        "  </head>",
        "</html>",
    ].join("\n");

    const patchedOnce = patchIndexHtml(original);
    const patchedTwice = patchIndexHtml(patchedOnce);

    assert.ok(patchedOnce.indexOf(INJECTED_SCRIPT_TAG) < patchedOnce.indexOf("index-example.js"));
    assert.equal(patchedTwice, patchedOnce);
    assert.equal(patchedTwice.split(INJECTED_SCRIPT_TAG).length - 1, 1);
});

test("installs idempotently, refreshes update backups, and restores", () => {
    const {
        installIntoExtension,
        restoreFromExtension,
    } = require("./patch-codex-webview-keyboard-tools");
    const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "codex-webview-tools-"));
    const extensionDirectory = path.join(temporaryRoot, "openai.chatgpt-test");
    const webviewDirectory = path.join(extensionDirectory, "webview");
    const sourceScript = path.join(temporaryRoot, "source.js");
    const indexPath = path.join(webviewDirectory, "index.html");
    const backupPath = `${indexPath}.codex-keyboard-tools.original`;
    const installedScriptPath = path.join(webviewDirectory, "codex-keyboard-tools.js");
    const makeIndex = (assetName) => [
        "<html>",
        "  <head>",
        `    <script type="module" crossorigin src="./assets/${assetName}"></script>`,
        "  </head>",
        "</html>",
    ].join("\n");
    try {
        fs.mkdirSync(webviewDirectory, { recursive: true });
        fs.writeFileSync(path.join(extensionDirectory, "package.json"), JSON.stringify({
            name: "chatgpt",
            publisher: "openai",
            version: "1.0.0",
        }));
        fs.writeFileSync(sourceScript, "source-v1\n");
        fs.writeFileSync(indexPath, makeIndex("index-v1.js"));

        installIntoExtension(extensionDirectory, "1.0.0", sourceScript);
        installIntoExtension(extensionDirectory, "1.0.0", sourceScript);
        assert.match(fs.readFileSync(indexPath, "utf8"), /codex-keyboard-tools\.js/);
        assert.equal(fs.readFileSync(backupPath, "utf8"), makeIndex("index-v1.js"));
        assert.equal(fs.readFileSync(installedScriptPath, "utf8"), "source-v1\n");

        fs.writeFileSync(sourceScript, "source-v2\n");
        fs.writeFileSync(indexPath, makeIndex("index-v2.js"));
        installIntoExtension(extensionDirectory, "1.0.0", sourceScript);
        assert.equal(fs.readFileSync(backupPath, "utf8"), makeIndex("index-v2.js"));
        assert.equal(fs.readFileSync(installedScriptPath, "utf8"), "source-v2\n");

        restoreFromExtension(extensionDirectory, "1.0.0");
        assert.equal(fs.readFileSync(indexPath, "utf8"), makeIndex("index-v2.js"));
        assert.equal(fs.existsSync(installedScriptPath), false);
    } finally {
        fs.rmSync(temporaryRoot, { force: true, recursive: true });
    }
});

test("keeps the public Ctrl+N new-chat binding singular", () => {
    const extensionManifest = JSON.parse(fs.readFileSync(
        "/home/barry/.vscode/extensions/openai.chatgpt-26.825.51511-linux-x64/package.json",
        "utf8",
    ));
    const newChatBindings = extensionManifest.contributes.keybindings.filter((binding) => (
        binding.command === "chatgpt.newChat" && binding.key === "ctrl+n"
    ));
    assert.equal(newChatBindings.length, 1, "public Ctrl+N new-chat binding count");

});
