"""Native long-lived Pipecat wiring; model/tokenizer are labelled fixtures."""
import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.session import AssistantSession
from test_mem0_adapter import actual_sdk
from test_pipecat_pipeline import FrameworkFakeLLM


class PersistentSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.memory = Mem0Adapter(root / "memory", sdk=actual_sdk(root / "memory"), embedding_scope="FIXTURE")
        self.records = Records(root / "records.sqlite")
        self.reminders = ReminderService(root / "reminders.sqlite")
        self.llm = FrameworkFakeLLM()
        self.session = AssistantSession(LlamaConfig("fixture"), self.memory, self.reminders, self.records, llm=self.llm)
        self.session.budget.count = AsyncMock(return_value=100)
        self.evidence = patch("app.session.memory_message", return_value=({"role": "user", "content": "FIXTURE []"},
            {"injected_memory_ids": [], "memory_tokens": 3, "retrieved_count": 0, "injected_count": 0}))
        self.evidence.start()
        await self.session.start()

    async def asyncTearDown(self):
        await self.session.close()
        self.evidence.stop()
        self.reminders.close(); self.records.close(); self.memory.close(); self.temp.cleanup()

    async def test_two_turns_use_one_worker_and_persist_only_after_output(self):
        emitted = []
        async def emit(text): emitted.append(text)
        worker = self.session.bot.worker
        await self.session.ask("你好", emit=emit)
        await self.session.ask("继续聊", emit=emit)
        self.assertIs(self.session.bot.worker, worker)
        self.assertFalse(worker.has_finished())
        self.assertEqual(self.llm.context_frames_seen, 2)
        self.assertEqual(len(emitted), 2)
        self.assertEqual(len(self.records.recent("owner")), 2)

    async def test_configured_retrieval_count_reaches_store_and_evidence_budget(self):
        await self.session.close();self.evidence.stop()
        # Equal fixture vectors isolate count/filter/budget wiring from model
        # quality, using the actual Mem0/Qdrant store and native session worker.
        self.memory.sdk.embedding_model.embed=lambda *_args,**_kwargs:[1.0]+[0.0]*31
        for i in range(9):
            self.memory.remember('owner','fact',f'bounded-{i}',f'合成资料{i}需要保留原文。',source_id=f's{i}',request_id=f'save-{i}')
        self.memory.remember('guest','fact','private','访客资料不得进入主人上下文。',source_id='guest',request_id='guest')
        self.session=AssistantSession(LlamaConfig('fixture'),self.memory,self.reminders,self.records,
            llm=FrameworkFakeLLM(),memory_top_k=8,memory_context_tokens=256)
        self.session.budget.count=AsyncMock(return_value=100)
        def tokenize(request,**_kwargs):
            text=json.loads(request.data)['content']
            return io.BytesIO(json.dumps({'tokens':list(range(len(text)))}).encode())
        async def emit(_text):pass
        with patch('app.context_memory.urlopen',tokenize):
            await self.session.start();result=await self.session.ask('请回忆这些合成资料。',emit=emit)
        self.assertEqual(result['retrieval']['retrieved_count'],8)
        self.assertEqual(result['retrieval']['requested_top_k'],8)
        self.assertEqual(result['retrieval']['evidence_token_limit'],256)
        self.assertLessEqual(result['retrieval']['memory_tokens'],256)
        self.assertTrue(result['retrieval']['injected_memory_ids'])
        self.assertLess(len(result['retrieval']['injected_memory_ids']),8)
        self.assertTrue(all(x['user_id']=='owner' for x in self.session.turn.memory_candidates))

    async def test_invalid_memory_limits_fail_before_pipeline_resources_exist(self):
        for kwargs in ({'memory_top_k':True},{'memory_top_k':0},{'memory_top_k':21},
                       {'memory_context_tokens':True},{'memory_context_tokens':0},{'memory_context_tokens':2048}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                AssistantSession(LlamaConfig('fixture'),self.memory,self.reminders,self.records,**kwargs)

    async def test_failed_output_is_not_completed_history(self):
        async def broken_output(_text): raise OSError("fixture output failure")
        with self.assertRaises(OSError): await self.session.ask("你好", emit=broken_output)
        self.assertEqual(self.records.recent("owner"), [])

    async def test_history_waits_for_emit_receipt_and_keeps_its_output_channel(self):
        self.session.output_channel = "fixture"
        entered, drained = asyncio.Event(), asyncio.Event()
        async def emit(_text):
            entered.set()
            await drained.wait()
        task = asyncio.create_task(self.session.ask("你好", emit=emit))
        await asyncio.wait_for(entered.wait(), 3)
        self.assertEqual(self.records.recent("owner"), [])
        drained.set()
        await task
        self.assertEqual(self.records.recent("owner")[0]["output_channel"], "fixture")

    async def test_cancel_after_model_completion_prevents_a_late_output_receipt(self):
        entered, drained = asyncio.Event(), asyncio.Event()
        async def emit(_text):
            entered.set()
            await drained.wait()
        task = asyncio.create_task(self.session.ask("你好", emit=emit))
        await asyncio.wait_for(entered.wait(), 3)
        self.assertTrue(self.session.pending.done())
        await self.session.cancel()
        drained.set()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(self.records.recent("owner"), [])

    async def test_empty_provider_observation_cannot_reopen_admission(self):
        entered,release=asyncio.Event(),asyncio.Event()
        async def emit(_text):entered.set();await release.wait()
        task=asyncio.create_task(self.session.ask('你好',emit=emit))
        await asyncio.wait_for(entered.wait(),3)
        self.session.real_model=True
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=[])))
        try:
            with patch('app.session.httpx.AsyncClient',return_value=client):
                with self.assertRaisesRegex(RuntimeError,'Invalid provider slot observation'):await self.session.cancel()
            self.assertFalse(self.session.backend_available)
            release.set();await asyncio.gather(task,return_exceptions=True)
            self.assertEqual(self.records.recent('owner'),[])
            with self.assertRaisesRegex(RuntimeError,'尚未确认'):await self.session.ask('继续',emit=emit)
        finally:
            release.set();await asyncio.gather(task,return_exceptions=True);await client.aclose()

    async def test_missing_time_uses_native_response_without_model_and_stop_clears_it(self):
        async def emit(_text): pass
        result = await self.session.ask("设置一个喝水提醒。", emit=emit)
        self.assertTrue(result["model_request_skipped"])
        self.assertEqual(self.llm.context_frames_seen, 0)
        self.assertIsNotNone(self.session.pending_clarification)
        await self.session.cancel()
        self.assertIsNone(self.session.pending_clarification)

    async def test_barge_in_can_fill_time_before_clarification_audio_finishes(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def emit(_text):
            entered.set()
            await release.wait()
        task = asyncio.create_task(self.session.ask("明天提醒我交报告。", emit=emit))
        await asyncio.wait_for(entered.wait(), 3)
        self.assertIsNotNone(self.session.pending_clarification)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertIsNotNone(self.session.pending_clarification)
        async def output(_text): pass
        await self.session.ask("上午八点", emit=output)
        self.assertIn("交报告", self.session.turn.user_text)
        due = self.session.turn.resolved_time.due
        self.assertIsNotNone(due)
        self.assertEqual(due.hour, 8)
        self.assertEqual((due.date()-self.session.turn.accepted_at.astimezone(due.tzinfo).date()).days, 1)

    async def test_oversized_protected_input_never_reaches_model(self):
        self.session.budget.count = AsyncMock(return_value=2000)
        async def emit(_text): self.fail("Oversized request must not produce output")
        with self.assertRaises(ValueError): await self.session.ask("当前问题不能被偷偷截断", emit=emit)
        self.assertEqual(self.llm.context_frames_seen, 0)
        self.assertEqual(self.records.recent("owner"), [])

    async def test_missing_tool_execution_repairs_only_once_then_reports_failure(self):
        emitted = []
        async def emit(text): emitted.append(text)
        result = await self.session.ask("请记住我喜欢绿茶。", emit=emit)
        self.assertTrue(result["repair_attempted"])
        self.assertEqual(self.llm.context_frames_seen, 2)
        self.assertEqual(len(emitted), 1)
        self.assertIn("无法确认", emitted[0])
        self.assertEqual(self.memory.list_memories("owner"), [])


if __name__ == "__main__": unittest.main()
