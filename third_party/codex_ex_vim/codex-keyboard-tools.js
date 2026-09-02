(function initializeCodexKeyboardTools(globalScope, factory) {
    "use strict";

    const api = factory();

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }

    if (globalScope?.document && typeof globalScope.addEventListener === "function") {
        api.install(globalScope);
    }
})(typeof globalThis === "undefined" ? this : globalThis, function createApi() {
    "use strict";

    const CODE_BLOCK_SELECTOR = '[data-markdown-copy="code-block"]';
    const TIMELINE_SELECTOR = "[data-pip-anchor-host]";
    const HIDDEN_SELECTOR = '[hidden], [aria-hidden="true"], [inert]';
    const ACTIVE_ATTRIBUTE = "data-codex-keyboard-active-code";
    const INSTALL_FLAG = "__codexKeyboardToolsInstalled";
    const LINE_SCROLL_PX = 160;
    const PAGE_SCROLL_RATIO = 0.8;

    function getCodeText(block) {
        if (!block) return "";

        const rawText = block.getAttribute?.("data-markdown-copy-text");
        if (rawText != null) return rawText;

        return block.querySelector?.("code")?.textContent ?? block.textContent ?? "";
    }

    function createController(windowObject, options = {}) {
        const documentObject = windowObject.document;
        const shouldShowStatus = options.showStatus !== false;
        let activeBlock = null;
        let activeBlockIndex = -1;
        let activeBlockKey = null;
        let revealGeneration = 0;
        let statusElement = null;
        let statusTimer = null;

        function isScrollable(element) {
            return element != null
                && Number(element.scrollHeight) > Number(element.clientHeight) + 1;
        }

        function findScrollableAncestor(element) {
            for (let current = element?.parentElement; current; current = current.parentElement) {
                if (isScrollable(current)) return current;
            }
            return null;
        }

        function getBlockScrollContainer(block) {
            const ancestor = findScrollableAncestor(block);
            if (ancestor) return ancestor;

            const visibleTimeline = getVisibleTimeline();
            return visibleTimeline?.contains?.(block) && isScrollable(visibleTimeline)
                ? visibleTimeline
                : null;
        }

        function getBlockKey(block) {
            const scope = block?.closest?.("[data-turn-key]")
                ?? block?.closest?.("[data-content-search-turn-key]")
                ?? block?.closest?.("[data-chatgpt-conversation-turn-id]")
                ?? block?.closest?.("[data-content-search-unit-key]");
            const scopeKey = scope?.getAttribute?.("data-turn-key")
                ?? scope?.getAttribute?.("data-content-search-turn-key")
                ?? scope?.getAttribute?.("data-chatgpt-conversation-turn-id")
                ?? scope?.getAttribute?.("data-content-search-unit-key");
            const code = block?.querySelector?.("[data-code-block-index]");
            let codeIndex = code?.getAttribute?.("data-code-block-index");
            if (codeIndex == null && scope?.querySelectorAll) {
                const ordinal = Array.from(scope.querySelectorAll(CODE_BLOCK_SELECTOR)).indexOf(block);
                if (ordinal >= 0) codeIndex = String(ordinal);
            }
            if (scopeKey == null || codeIndex == null) return null;
            return `${scopeKey}\u0000${codeIndex}`;
        }

        function findBlockByKey(blockKey) {
            if (blockKey == null) return null;
            return Array.from(documentObject.querySelectorAll(CODE_BLOCK_SELECTOR))
                .find((block) => getBlockKey(block) === blockKey) ?? null;
        }

        function largestScrollable(elements) {
            return Array.from(elements)
                .filter(isScrollable)
                .sort((left, right) => {
                    const leftRange = left.scrollHeight - left.clientHeight;
                    const rightRange = right.scrollHeight - right.clientHeight;
                    return rightRange - leftRange;
                })[0] ?? null;
        }

        function largestByScrollRange(elements) {
            return Array.from(elements)
                .sort((left, right) => {
                    const leftRange = left.scrollHeight - left.clientHeight;
                    const rightRange = right.scrollHeight - right.clientHeight;
                    return rightRange - leftRange;
                })[0] ?? null;
        }

        function hasVisibleLayout(element) {
            if (element?.isConnected === false || element?.closest?.(HIDDEN_SELECTOR)) return false;
            const rect = element?.getBoundingClientRect?.();
            return rect != null && Number(rect.width) > 0 && Number(rect.height) > 0;
        }

        function getVisibleTimeline() {
            const timelines = Array.from(documentObject.querySelectorAll(TIMELINE_SELECTOR))
                .filter(hasVisibleLayout);
            const focusedTimeline = timelines.find((timeline) => (
                documentObject.activeElement != null
                && timeline.contains?.(documentObject.activeElement)
            ));
            return focusedTimeline ?? largestByScrollRange(timelines);
        }

        function getCodeBlocks() {
            const timeline = getVisibleTimeline();
            return Array.from(documentObject.querySelectorAll(CODE_BLOCK_SELECTOR))
                .filter(hasVisibleLayout)
                .filter((block) => timeline == null || timeline.contains?.(block));
        }

        function getScrollContainer() {
            const timeline = getVisibleTimeline();
            if (timeline && isScrollable(timeline)) return timeline;

            if (hasVisibleLayout(activeBlock)) {
                const ancestor = findScrollableAncestor(activeBlock);
                if (ancestor && hasVisibleLayout(ancestor)) return ancestor;
            }

            return largestScrollable(
                Array.from(documentObject.querySelectorAll("div, main, section"))
                    .filter(hasVisibleLayout),
            );
        }

        function showStatus(message) {
            if (!shouldShowStatus || !documentObject.body?.appendChild) return;

            if (!statusElement) {
                statusElement = documentObject.createElement("div");
                statusElement.id = "codex-keyboard-tools-status";
                statusElement.setAttribute("role", "status");
                statusElement.style.cssText = [
                    "position:fixed",
                    "top:8px",
                    "right:8px",
                    "z-index:2147483647",
                    "max-width:calc(100% - 16px)",
                    "padding:5px 8px",
                    "border:1px solid rgba(127,127,127,.45)",
                    "border-radius:4px",
                    "background:var(--color-surface-primary, #202020)",
                    "color:var(--color-text-primary, #f5f5f5)",
                    "font:12px/1.4 sans-serif",
                    "box-shadow:0 2px 8px rgba(0,0,0,.24)",
                    "pointer-events:none",
                ].join(";");
                documentObject.body.appendChild(statusElement);
            }

            statusElement.textContent = message;
            statusElement.hidden = false;
            if (statusTimer != null) windowObject.clearTimeout?.(statusTimer);
            statusTimer = windowObject.setTimeout?.(() => {
                statusElement.hidden = true;
            }, 1400) ?? null;
        }

        function nearestBlockIndex(blocks) {
            if (blocks.length === 0) return -1;

            const viewportCenter = Number(windowObject.innerHeight || 0) / 2;
            let bestIndex = 0;
            let bestDistance = Number.POSITIVE_INFINITY;

            blocks.forEach((block, index) => {
                const rect = block.getBoundingClientRect?.();
                if (!rect) return;
                const blockCenter = Number(rect.top) + Number(rect.height || 0) / 2;
                const distance = Math.abs(blockCenter - viewportCenter);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestIndex = index;
                }
            });

            return bestIndex;
        }

        function revealDelta(block, timeline) {
            const blockRect = block.getBoundingClientRect?.();
            const timelineRect = timeline.getBoundingClientRect?.();
            if (!blockRect || !timelineRect) return 0;

            const timelineHeight = Number(timelineRect.height || timeline.clientHeight || 0);
            if (timelineHeight <= 0) return 0;

            const topInset = Math.min(24, timelineHeight / 4);
            const configuredBottomPadding = Number.parseFloat(
                timeline.style?.getPropertyValue?.("--thread-scroll-padding-bottom") ?? "",
            );
            const bottomPadding = Number.isFinite(configuredBottomPadding)
                ? configuredBottomPadding
                : 0;
            const bottomInset = Math.min(
                Math.max(24, bottomPadding),
                Math.max(24, timelineHeight * 0.45),
            );
            const visibleTop = Number(timelineRect.top) + topInset;
            const visibleBottom = Number(timelineRect.bottom) - bottomInset;
            const visibleHeight = Math.max(0, visibleBottom - visibleTop);

            if (Number(blockRect.height) >= visibleHeight) {
                return Number(blockRect.top) - visibleTop;
            }
            if (Number(blockRect.top) < visibleTop) {
                return Number(blockRect.top) - visibleTop;
            }
            if (Number(blockRect.bottom) > visibleBottom) {
                return Number(blockRect.bottom) - visibleBottom;
            }
            return 0;
        }

        function revealBlock(block) {
            const timeline = getBlockScrollContainer(block);
            if (!timeline || !timeline.contains?.(block) || !hasVisibleLayout(block)) return false;

            const generation = ++revealGeneration;
            const blockKey = getBlockKey(block);
            const adjust = () => {
                let currentBlock = block;
                if (!hasVisibleLayout(currentBlock)) {
                    currentBlock = findBlockByKey(blockKey);
                }
                if (generation !== revealGeneration
                    || currentBlock == null
                    || !hasVisibleLayout(currentBlock)
                    || activeBlockKey !== blockKey && activeBlock !== currentBlock) {
                    return;
                }

                if (activeBlock !== currentBlock && activeBlockKey === blockKey) {
                    activeBlock = currentBlock;
                    currentBlock.setAttribute?.(ACTIVE_ATTRIBUTE, "true");
                    currentBlock.setAttribute?.("tabindex", "-1");
                }

                const currentTimeline = getBlockScrollContainer(currentBlock);
                if (!currentTimeline || !currentTimeline.contains?.(currentBlock)) return;

                const delta = revealDelta(currentBlock, currentTimeline);
                if (Math.abs(delta) < 1) return;
                if (typeof currentTimeline.scrollBy === "function") {
                    currentTimeline.scrollBy({ behavior: "instant", left: 0, top: delta });
                } else {
                    currentTimeline.scrollTop += delta;
                }
            };

            adjust();
            windowObject.requestAnimationFrame?.(adjust);
            return true;
        }

        function activateBlock(blocks, index) {
            blocks.forEach((block) => block.removeAttribute?.(ACTIVE_ATTRIBUTE));
            const block = blocks[index];
            if (!block) return null;

            activeBlock = block;
            activeBlockIndex = index;
            activeBlockKey = getBlockKey(block);
            block.setAttribute?.(ACTIVE_ATTRIBUTE, "true");
            block.setAttribute?.("tabindex", "-1");
            revealBlock(block);
            block.focus?.({ preventScroll: true });
            showStatus(`Code block ${index + 1}/${blocks.length}`);
            return block;
        }

        function selectCodeBlock(direction) {
            const blocks = getCodeBlocks();
            if (blocks.length === 0) {
                activeBlock = null;
                // Keep the logical selection through transient virtualization gaps.
                showStatus("No code block found");
                return null;
            }

            const currentIndex = blocks.indexOf(activeBlock);
            const keyedIndex = currentIndex === -1 && activeBlockKey != null
                ? blocks.findIndex((block) => getBlockKey(block) === activeBlockKey)
                : -1;
            const rememberedIndex = currentIndex !== -1
                ? currentIndex
                : keyedIndex !== -1
                    ? keyedIndex
                    : activeBlockIndex;
            const hasRememberedIndex = rememberedIndex >= 0 && rememberedIndex < blocks.length;
            const nextIndex = !hasRememberedIndex
                ? nearestBlockIndex(blocks)
                : (rememberedIndex + direction + blocks.length) % blocks.length;

            return activateBlock(blocks, nextIndex);
        }

        function ensureActiveBlock() {
            const blocks = getCodeBlocks();
            if (blocks.includes(activeBlock)) return activeBlock;
            if (blocks.length === 0) return null;
            const keyedIndex = activeBlockKey == null
                ? -1
                : blocks.findIndex((block) => getBlockKey(block) === activeBlockKey);
            const hasRememberedIndex = keyedIndex !== -1
                || activeBlockIndex >= 0 && activeBlockIndex < blocks.length;
            const index = keyedIndex !== -1
                ? keyedIndex
                : hasRememberedIndex
                    ? activeBlockIndex
                    : nearestBlockIndex(blocks);
            return activateBlock(blocks, index);
        }

        async function writeClipboard(text) {
            if (windowObject.navigator?.clipboard?.writeText) {
                await windowObject.navigator.clipboard.writeText(text);
                return true;
            }

            if (!documentObject.createElement || !documentObject.body?.appendChild) return false;
            const textarea = documentObject.createElement("textarea");
            textarea.value = text;
            textarea.style.position = "fixed";
            textarea.style.opacity = "0";
            documentObject.body.appendChild(textarea);
            textarea.select();
            const copied = documentObject.execCommand?.("copy") === true;
            textarea.remove();
            return copied;
        }

        async function copyActiveCode() {
            const block = ensureActiveBlock();
            if (!block) {
                showStatus("No code block found");
                return false;
            }

            const text = getCodeText(block);
            const copied = await writeClipboard(text);
            showStatus(copied ? "Code block copied" : "Copy failed");
            return copied;
        }

        function scrollByPixels(delta) {
            const container = getScrollContainer();
            if (!container) {
                showStatus("Conversation scroller not found");
                return false;
            }

            if (typeof container.scrollBy === "function") {
                container.scrollBy({ behavior: "instant", left: 0, top: delta });
            } else {
                container.scrollTop += delta;
            }
            return true;
        }

        function scrollByPage(direction) {
            const container = getScrollContainer();
            if (!container) {
                showStatus("Conversation scroller not found");
                return false;
            }

            const pageSize = container.clientHeight > 0
                ? Math.max(1, Math.round(container.clientHeight * PAGE_SCROLL_RATIO))
                : LINE_SCROLL_PX;
            return scrollByPixels(pageSize * direction);
        }

        function jumpLatest() {
            const container = getScrollContainer();
            if (!container) {
                showStatus("Conversation scroller not found");
                return false;
            }

            if (typeof container.scrollTo === "function") {
                container.scrollTo({ behavior: "instant", left: 0, top: 0 });
            } else {
                container.scrollTop = 0;
            }
            showStatus("Latest message");
            return true;
        }

        function jumpOldest() {
            const container = getScrollContainer();
            if (!container) {
                showStatus("Conversation scroller not found");
                return false;
            }

            const oldestPosition = Math.min(0, container.clientHeight - container.scrollHeight);
            if (typeof container.scrollTo === "function") {
                container.scrollTo({ behavior: "instant", left: 0, top: oldestPosition });
            } else {
                container.scrollTop = oldestPosition;
            }
            showStatus("Oldest message");
            return true;
        }

        function handleKeydown(event) {
            if (event.repeat || event.metaKey || event.shiftKey) return false;
            const key = String(event.key).toLowerCase();
            let localMessageType = null;
            if (event.ctrlKey && event.altKey && key === "s") {
                localMessageType = "chat-search-command-menu";
            } else if (event.ctrlKey && !event.altKey && key === "[") {
                localMessageType = "navigate-back";
            }
            if (localMessageType != null) {
                if (typeof windowObject.postMessage !== "function") return false;
                event.preventDefault();
                event.stopImmediatePropagation();
                windowObject.postMessage(
                    { type: localMessageType },
                    windowObject.location?.origin || "*",
                );
                return true;
            }

            if (!event.altKey || event.ctrlKey) return false;
            const actions = {
                c: () => {
                    copyActiveCode().catch(() => showStatus("Copy failed"));
                },
                d: () => scrollByPage(1),
                end: jumpLatest,
                g: jumpLatest,
                home: jumpOldest,
                j: () => scrollByPixels(LINE_SCROLL_PX),
                k: () => scrollByPixels(-LINE_SCROLL_PX),
                n: () => selectCodeBlock(1),
                p: () => selectCodeBlock(-1),
                u: () => scrollByPage(-1),
            };
            const action = actions[key];
            if (!action) return false;

            event.preventDefault();
            // Let Codex's timeline keydown cleanup run after suppressing browser defaults.
            action();
            return true;
        }

        return {
            copyActiveCode,
            getActiveBlock: () => activeBlock,
            getScrollContainer,
            handleKeydown,
            jumpLatest,
            jumpOldest,
            scrollByPage,
            scrollByPixels,
            selectCodeBlock,
        };
    }

    function install(windowObject) {
        if (windowObject[INSTALL_FLAG]) return windowObject[INSTALL_FLAG];

        const style = windowObject.document.createElement("style");
        style.id = "codex-keyboard-tools-style";
        style.textContent = [
            `[${ACTIVE_ATTRIBUTE}="true"] {`,
            "outline: 2px solid var(--vscode-focusBorder, #3b82f6) !important;",
            "outline-offset: 2px !important;",
            "}",
        ].join("\n");
        windowObject.document.head?.appendChild(style);

        const controller = createController(windowObject);
        windowObject.addEventListener("keydown", controller.handleKeydown, true);
        windowObject[INSTALL_FLAG] = controller;
        return controller;
    }

    return {
        createController,
        getCodeText,
        install,
    };
});
