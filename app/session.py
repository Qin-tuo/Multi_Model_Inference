"""Persistent text transport built around one native Pipecat worker/context.

This module supplies input and output callbacks. Pipecat owns model generation,
tool continuation, context aggregation, interruption, and worker shutdown.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid

import httpx
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (EndFrame, ErrorFrame, InterruptionFrame, LLMContextFrame,
    LLMMessagesAppendFrame, LLMMessagesUpdateFrame, LLMSetToolChoiceFrame, LLMSetToolsFrame)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from app.context_memory import memory_message
from app.pipecat_bot import SYSTEM_MESSAGE, build_pipeline
from app.tools import ToolResponseGuard, ToolTurn


class PromptBudget(FrameProcessor):
    """Measure the provider-rendered prompt before every native LLM request."""
    def __init__(self, base_url, *, context_tokens, output_tokens):
        super().__init__()
        self.base = base_url.removesuffix("/").removesuffix("/v1")
        self.context_tokens = context_tokens
        self.output_tokens = output_tokens
        self.client = httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=1))
        self.adapter = OpenAILLMAdapter()
        self.measurements = []

    async def count(self, context):
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)
        body = {key: value for key, value in params.items() if isinstance(value, (dict, list, str))}
        rendered = await self.client.post(self.base + "/apply-template", json=body)
        rendered.raise_for_status()
        tokens = await self.client.post(self.base + "/tokenize", json={
            "content": rendered.json()["prompt"], "add_special": True, "parse_special": True})
        tokens.raise_for_status()
        return len(tokens.json()["tokens"])

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            if not await self.check(frame.context): return
        await self.push_frame(frame, direction)

    async def check(self, context):
        try:
            count = await self.count(context)
            self.measurements.append(count)
            if count + self.output_tokens + 16 > self.context_tokens:
                raise ValueError("CONTEXT_BUDGET_EXCEEDED: current request and tool results exceed model capacity")
            return True
        except (httpx.HTTPError, ValueError, KeyError) as error:
            await self.push_error_frame(ErrorFrame(error=str(error)))
            return False

    async def cleanup(self):
        await self.client.aclose()
        await super().cleanup()


class ToolContinuationBudget(FrameProcessor):
    """Tool results reach the service upstream, so check that path as well."""
    def __init__(self, budget):
        super().__init__()
        self.budget = budget

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.UPSTREAM:
            if not await self.budget.check(frame.context): return
        await self.push_frame(frame, direction)


class AssistantSession:
    def __init__(self, config, memory, reminders, records, *, user_id="owner", context_tokens=2048,
                 session_id=None, llm=None):
        self.config, self.memory, self.reminders, self.records = config, memory, reminders, records
        self.user_id = user_id
        self.session_id = session_id or uuid.uuid4().hex
        self.turn = ToolTurn(user_id, "", memory, reminders, records=records)
        self.guard = ToolResponseGuard(self.turn)
        self.budget = PromptBudget(config.base_url, context_tokens=context_tokens, output_tokens=config.max_tokens)
        self.bot = build_pipeline(config, [{"role": "system", "content": SYSTEM_MESSAGE}], llm=llm,
                                  before_llm=[self.budget], after_llm=[ToolContinuationBudget(self.budget), self.guard], conversation_id=self.session_id)
        self.runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        self.ready = asyncio.Event()
        self.pending = None
        self.output = None
        self.run_task = None
        self.busy = asyncio.Lock()
        self.retrieval = {}
        self.cancel_task = None
        self.backend_available = True
        self.repair_attempted = False
        self.real_model = llm is None

        @self.bot.worker.event_handler("on_pipeline_started")
        async def started(_worker, _frame):
            self.ready.set()

        @self.bot.aggregators.assistant().event_handler("on_assistant_turn_stopped")
        async def stopped(_aggregator, message):
            if not self.pending or self.pending.done(): return
            if message.interrupted or self.turn.cancelled:
                self.pending.set_exception(asyncio.CancelledError())
            elif message.content.strip():
                if self.turn.response_error() and not self.repair_attempted:
                    self.repair_attempted = True
                    required = self.turn.required_operation()
                    self.bot.context.set_tool_choice("required")
                    await self.bot.worker.queue_frame(LLMMessagesAppendFrame(messages=[{
                        "role": "user", "content": f"上一轮没有获得 {required} 的成功结果。请修复工具调用，先输出 <tool_call> 开始的调用，禁止先写说明文字。原请求：{self.turn.user_text}。" + self.turn.system_guidance()
                    }], run_llm=True))
                    return
                self.pending.set_result(message.content)

        @self.bot.worker.event_handler("on_pipeline_error")
        async def error(_worker, frame):
            if self.pending and not self.pending.done():
                self.pending.set_exception(RuntimeError(frame.error))

    async def start(self):
        if self.real_model:
            health = await self.budget.client.get(self.budget.base + "/health")
            health.raise_for_status()
            if health.json().get("status") != "ok":
                raise RuntimeError("Local llama.cpp model is not ready")
        await self.runner.add_workers(self.bot.worker)
        self.run_task = asyncio.create_task(self.runner.run())
        await asyncio.wait_for(self.ready.wait(), 10)
        return self

    async def close(self):
        if self.run_task:
            if self.pending and not self.pending.done():
                await self.cancel()
            if not self.bot.worker.has_finished():
                await self.bot.worker.queue_frame(EndFrame())
            await asyncio.wait_for(self.run_task, 15)
            self.run_task = None

    async def cancel(self):
        """Native interruption plus observed provider slot completion.

        This is logical text cancellation, not physical playback/AEC validation.
        """
        if self.cancel_task is not None:
            await asyncio.shield(self.cancel_task)
            return
        if not self.pending or self.pending.done(): return
        self.cancel_task = asyncio.create_task(self._cancel_backend())
        await asyncio.shield(self.cancel_task)

    async def _cancel_backend(self):
        self.backend_available = False
        self.turn.cancel()
        await self.bot.worker.queue_frame(InterruptionFrame())
        if not self.pending.done():
            self.pending.set_exception(asyncio.CancelledError())
        # Handler writes already admitted may complete; hold admission until
        # their shielded SDK work has returned, then inspect provider state.
        async with asyncio.timeout(15):
            async with self.turn._gate:
                pass
        deadline = time.monotonic() + 10
        async with httpx.AsyncClient(timeout=2) as client:
            while time.monotonic() < deadline:
                result = await client.get(self.budget.base + "/slots")
                result.raise_for_status()
                if all(not slot.get("is_processing", True) for slot in result.json()):
                    self.backend_available = True
                    return
                await asyncio.sleep(0.05)
        raise RuntimeError("Backend is still processing the cancelled request")

    async def _messages(self, text, turn):
        task_lookup = turn.requested_lookup() in {"list_notes", "list_reminders", "get_time"} and turn.required_operation() == turn.requested_lookup()
        # Notes/task state have their own authoritative store. Irrelevant
        # semantic matches must not be presented as evidence for those queries.
        found = ({"items": [], "generation": self.memory.generation(self.user_id)} if task_lookup else
                 await asyncio.to_thread(self.memory.search, self.user_id, text, top_k=5))
        found["items"] = self.records.filter_memories(self.user_id, found["items"])
        evidence, metrics = await asyncio.to_thread(memory_message, found["items"], self.config.base_url)
        self.records.attach_memories(self.user_id, turn.turn_id, metrics["injected_memory_ids"])
        history = self.records.recent(self.user_id, session_id=self.session_id, limit=8)
        # Explicit cross-session history questions use dated, delivered records.
        if any(word in text for word in ("上次", "之前聊", "之前讨论")):
            history = self.records.recent(self.user_id, limit=5)
        if task_lookup: history = []
        personal_recall = (re.search(r"我|我的|记得", text) and re.search(r"喜欢|偏好|姓名|名字|称呼|项目背景", text)
                           and re.search(r"什么|哪|吗|记得|知道|怎样", text))
        if personal_recall:
            # Shopping notes and prior generated answers are not authoritative
            # preference records, even when their wording is semantically close.
            history = []
        # Propagate evidence dependencies through history-derived replies, so a
        # later forget also invalidates those downstream transcripts.
        inherited = self.records.memory_refs(self.user_id, [row["id"] for row in history])
        self.records.attach_memories(self.user_id, turn.turn_id, inherited)
        pairs = [[{"role": "user", "content": row["user_text"]},
                  {"role": "assistant", "content": row["assistant_text"]}] for row in history]
        system = {"role": "system", "content": SYSTEM_MESSAGE + turn.system_guidance()}
        schemas = turn.schemas()
        reserve = 512
        dropped = 0
        while True:
            messages = [system, *[message for pair in pairs for message in pair], evidence,
                        {"role": "user", "content": text}]
            probe = LLMContext(messages=messages, tools=schemas, tool_choice=turn.initial_tool_choice())
            tokens = await self.budget.count(probe)
            if tokens + self.config.max_tokens + reserve <= self.budget.context_tokens:
                break
            if pairs:
                pairs.pop(0); dropped += 1
            elif evidence["content"].endswith("[]"):
                raise ValueError("问题与工具定义超过当前模型上下文容量，请缩短问题。")
            else:
                evidence, metrics = await asyncio.to_thread(memory_message, [], self.config.base_url)
        self.retrieval = {**metrics, "history_turns_dropped": dropped, "initial_prompt_tokens": tokens,
                          "memory_generation": found["generation"], "personal_recall": bool(personal_recall),
                          "task_lookup": task_lookup}
        return messages, schemas

    async def ask(self, text, *, emit, timeout=90):
        if self.busy.locked():
            raise RuntimeError("当前轮仍在处理，请先停止或等待完成。")
        async with self.busy:
            if not self.run_task or self.run_task.done():
                raise RuntimeError("Pipecat worker is not running")
            if not self.backend_available:
                raise RuntimeError("尚未确认上次请求停止，请检查或重启模型服务。")
            self.cancel_task = None
            self.repair_attempted = False
            turn = ToolTurn(self.user_id, text, self.memory, self.reminders, records=self.records)
            self.turn = self.guard.turn = turn
            self.records.begin(self.user_id, self.session_id, turn.turn_id, text)
            started = time.monotonic()
            self.budget.measurements.clear()
            try:
                messages, schemas = await self._messages(text, turn)
                self.pending = asyncio.get_running_loop().create_future()
                await self.bot.worker.queue_frames([
                    LLMSetToolsFrame(schemas), LLMSetToolChoiceFrame(turn.initial_tool_choice()),
                    LLMMessagesUpdateFrame(messages=messages, run_llm=True)])
                answer = await asyncio.wait_for(asyncio.shield(self.pending), timeout)
                if turn.cancelled: raise asyncio.CancelledError()
                await emit(answer)
                persisted = await asyncio.to_thread(self.records.finish, self.user_id, turn.turn_id, answer)
                return {"turn_id": turn.turn_id, "session_id": self.session_id, "response": answer,
                        "calls": turn.calls, "retrieval": self.retrieval, "prompt_tokens": list(self.budget.measurements),
                        "history_persisted": persisted, "repair_attempted": self.repair_attempted,
                        "elapsed_seconds": time.monotonic() - started}
            except (TimeoutError, asyncio.CancelledError):
                await self.cancel()
                self.records.finish(self.user_id, turn.turn_id, "", status="cancelled")
                raise
            except Exception:
                self.records.finish(self.user_id, turn.turn_id, "", status="failed")
                raise
            finally:
                self.pending = None
