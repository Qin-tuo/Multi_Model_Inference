"""Real SDK state and native Pipecat output-policy tests; fixture embeddings."""
import asyncio
from datetime import timedelta
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.pipecat_bot import input_messages, run_smoke
from app.reminders import ReminderService
from app.records import Records
from app.tools import Args, CancelReminderArgs, CreateReminderArgs, ForgetArgs, RememberArgs, ToolTurn
from test_mem0_adapter import actual_sdk
from test_pipecat_pipeline import FrameworkFakeLLM


class ToolFakeLLM(FrameworkFakeLLM):
    def __init__(self):
        super().__init__()
        self._register_event_handler("on_function_calls_started")


class ToolHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.memory = Mem0Adapter(root / "memory", sdk=actual_sdk(root / "memory"), embedding_scope="FIXTURE")
        self.reminders = ReminderService(root / "reminders.sqlite")

    def tearDown(self):
        self.reminders.close()
        self.memory.close()
        self.temp.cleanup()

    def turn(self, text="请记住我喜欢绿茶。"):
        return ToolTurn("owner", text, self.memory, self.reminders)

    async def test_durable_save_and_replay_have_one_record(self):
        turn = self.turn()
        args = {"key": "drink", "content": "用户喜欢绿茶"}
        self.assertIsNotNone(turn.response_error())
        first = await turn.execute("remember", RememberArgs, args)
        second = await turn.execute("remember", RememberArgs, args)
        self.assertTrue(first["ok"])
        self.assertEqual(first, second)
        self.assertEqual(len(self.memory.list_memories("owner")), 1)
        self.assertEqual(self.memory.list_memories("guest"), [])
        self.assertIsNone(turn.response_error())
        self.assertEqual(len(turn.calls), 2)

    async def test_chat_cannot_write_and_model_cannot_supply_owner(self):
        turn = self.turn("你好，今天聊点什么？")
        result = await turn.execute("remember", RememberArgs, {"key": "drink", "content": "虚构事实"})
        self.assertEqual(result["code"], "EXPLICIT_REQUEST_REQUIRED")
        allowed = self.turn()
        result = await allowed.execute("remember", RememberArgs, {"key": "drink", "content": "绿茶", "user_id": "guest"})
        self.assertEqual(result["code"], "INVALID_ARGUMENT")
        self.assertEqual(self.memory.list_memories("owner"), [])
        self.assertEqual(len(turn.calls), 1)

    async def test_relative_reminder_is_anchored_and_replay_is_deduplicated(self):
        turn = self.turn("五分钟后提醒我喝水。")
        args = {"text": "喝水", "delay_seconds": 300}
        result = await turn.execute("create_reminder", CreateReminderArgs, args)
        self.assertTrue(result["ok"])
        self.assertEqual(result, await turn.execute("create_reminder", CreateReminderArgs, args))
        self.assertEqual(result["result"]["due_utc"], (turn.accepted_at + timedelta(seconds=300)).isoformat())
        self.assertEqual(len(self.reminders.list_reminders("owner")), 1)
        self.assertIsNone(turn.response_error())

    async def test_invalid_time_is_not_confirmed(self):
        turn = self.turn("设置提醒：喝水。")
        result = await turn.execute("create_reminder", CreateReminderArgs, {"text": "喝水", "due_at": "2030-01-01T12:00:00"})
        self.assertFalse(result["ok"])
        self.assertIsNotNone(turn.response_error())
        self.assertEqual(self.reminders.list_reminders("owner"), [])

    async def test_negated_cancel_and_discussion_of_forgetting_have_no_effect(self):
        create = self.turn("五分钟后提醒我喝水。")
        result = await create.execute("create_reminder", CreateReminderArgs, {"text": "喝水", "delay_seconds": 300})
        reminder_id = result["result"]["reminder_id"]
        for text in ("请不要取消我的提醒。", "不用取消提醒。", "请解释五分钟后提醒是什么意思。"):
            turn = self.turn(text)
            refused = await turn.execute("cancel_reminder", CancelReminderArgs, {"reminder_id": reminder_id})
            self.assertEqual(refused["code"], "EXPLICIT_REQUEST_REQUIRED")
        self.assertEqual(self.reminders.list_reminders("owner")[0]["status"], "active")
        turn = self.turn("我忘记昨天说过什么了。")
        refused = await turn.execute("forget", ForgetArgs, {"memory_id": "unknown"})
        self.assertEqual(refused["code"], "EXPLICIT_REQUEST_REQUIRED")

    async def test_reminder_words_inside_a_note_do_not_schedule_a_reminder(self):
        from app.tools import CreateNoteArgs
        with Records(Path(self.temp.name) / "records.sqlite") as records:
            turn = ToolTurn("owner", "创建笔记，内容是五分钟后提醒我休息。", self.memory, self.reminders, records=records)
            result = await turn.execute("create_reminder", CreateReminderArgs, {"text": "休息", "delay_seconds": 300})
            self.assertEqual(result["code"], "EXPLICIT_REQUEST_REQUIRED")
            result = await turn.execute("create_note", CreateNoteArgs, {"title": "摘录", "body": "五分钟后提醒我休息。"})
            self.assertTrue(result["ok"])
            self.assertEqual(self.reminders.list_reminders("owner"), [])
            self.assertEqual(len(records.list_notes("owner")), 1)

    async def test_cancel_before_admission_and_bounded_attempts(self):
        turn = self.turn()
        turn.cancel()
        result = await turn.execute("remember", RememberArgs, {"key": "drink", "content": "绿茶"})
        self.assertEqual(result["code"], "CANCELLED")
        self.assertEqual(self.memory.list_memories("owner"), [])
        turn = self.turn("现在几点？")
        for _ in range(3):
            self.assertTrue((await turn.execute("get_time", Args, {}))["ok"])
        self.assertEqual((await turn.execute("get_time", Args, {}))["code"], "TOOL_LIMIT")
        self.assertEqual(len(turn.calls), 4)

    async def test_cancel_during_accepted_write_reports_actual_durable_state(self):
        turn = self.turn()
        entered, release = threading.Event(), threading.Event()
        invoke = turn._invoke

        def delayed(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test release timeout")
            return invoke(*args)

        with patch.object(turn, "_invoke", side_effect=delayed):
            task = asyncio.create_task(turn.execute("remember", RememberArgs, {"key": "drink", "content": "绿茶"}))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                await asyncio.sleep(0)
            finally:
                release.set()
            result = await task
        self.assertTrue(result["ok"])
        self.assertTrue(turn.cancelled)
        self.assertEqual(len(self.memory.list_memories("owner")), 1)
        self.assertIsNotNone(turn.response_error())

    async def test_native_pipeline_blocks_unexecuted_write_confirmation(self):
        turn = self.turn()
        result = await run_smoke(LlamaConfig("fixture"), input_messages(turn.user_text),
                                 llm=ToolFakeLLM(), tool_turn=turn, timeout_seconds=10)
        self.assertEqual(result["response"], turn.response_error())
        self.assertNotIn("FRAMEWORK_TEST_ONLY", result["response"])
        self.assertEqual(self.memory.list_memories("owner"), [])


if __name__ == "__main__":
    unittest.main()
