"""Actual SDK guards and mature temporal parsing, with fixture memory vectors."""
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.temporal import resolve_reminder_time
from app.tools import ToolTurn, CreateReminderArgs, RememberArgs, CreateNoteArgs, ForgetArgs, CancelReminderArgs
from test_mem0_adapter import actual_sdk


class TrustedToolPolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.memory = Mem0Adapter(root / "memory", sdk=actual_sdk(root / "memory"), embedding_scope="FIXTURE")
        self.records = Records(root / "records.sqlite")
        self.reminders = ReminderService(root / "reminders.sqlite")

    def tearDown(self):
        self.memory.close(); self.records.close(); self.reminders.close(); self.temp.cleanup()

    def turn(self, text): return ToolTurn("owner", text, self.memory, self.reminders, records=self.records)

    async def test_negated_or_withdrawn_requests_never_write(self):
        for text in ["十分钟后不要提醒我喝水。", "今天不要提醒我开会。", "设置十分钟后的喝水提醒。取消刚才的请求。"]:
            result = await self.turn(text).execute("create_reminder", CreateReminderArgs, {"text":"喝水", "delay_seconds":600})
            self.assertFalse(result["ok"])
        result = await self.turn("请记住我喜欢绿茶。算了，不用记了。").execute("remember", RememberArgs, {"key":"drink","content":"喜欢绿茶"})
        self.assertFalse(result["ok"])
        self.assertEqual(self.memory.list_memories("owner"), [])
        self.assertEqual(self.reminders.list_reminders("owner"), [])

    async def test_task_content_can_contain_cancel_without_becoming_a_cancellation(self):
        turn = self.turn("75分钟后提醒我取消会议。")
        result = await turn.execute("create_reminder", CreateReminderArgs, {"text":"取消会议", "delay_seconds":75})
        self.assertTrue(result["ok"])
        self.assertEqual(datetime.fromisoformat(result["result"]["due_utc"]), turn.accepted_at + timedelta(minutes=75))

    async def test_no_time_means_no_invented_schedule(self):
        turn = self.turn("明天提醒我交报告。")
        result = await turn.execute("create_reminder", CreateReminderArgs, {"text":"交报告", "delay_seconds":10})
        self.assertEqual(result["code"], "NEEDS_CLARIFICATION")
        self.assertIn("具体", turn.fixed_response)
        self.assertEqual(self.reminders.list_reminders("owner"), [])

    async def test_literal_fact_preserves_qualifiers_the_model_omits(self):
        turn = self.turn("请记下，我通常晚上九点看书。")
        result = await turn.execute("remember", RememberArgs, {"key":"reading","content":"晚上九点"})
        self.assertTrue(result["ok"])
        saved = self.memory.list_memories("owner")[0]["content"]
        self.assertIn("通常", saved)
        self.assertIn("看书", saved)

    async def test_explicit_note_fields_override_a_confused_model_title(self):
        turn = self.turn("创建笔记，标题是原话，内容是“不要删除我的提醒”。")
        result = await turn.execute("create_note", CreateNoteArgs, {"title":"不要删除我的提醒","body":"wrong"})
        self.assertTrue(result["ok"])
        note = self.records.list_notes("owner")[0]
        self.assertEqual(note["title"], "原话")
        self.assertEqual(note["body"], "不要删除我的提醒")

    async def test_vague_destructive_reference_requires_clarification(self):
        for i, content in enumerate(["喜欢绿茶", "喜欢散步"]):
            self.memory.remember("owner","fact",str(i),content,source_id=str(i),request_id=str(i))
        target = self.memory.list_memories("owner")[0]["memory_id"]
        result = await self.turn("忘记那件事。").execute("forget", ForgetArgs, {"memory_id":target})
        self.assertEqual(result["code"], "NEEDS_CLARIFICATION")
        self.assertEqual(len(self.memory.list_memories("owner")), 2)

    def test_duration_arithmetic_is_anchored_and_does_not_use_model_numbers(self):
        anchor = datetime(2026,10,13,4,30,0,123456,tzinfo=ZoneInfo("Asia/Shanghai"))
        for expression, seconds in [("一个半小时",5400),("四十五分钟",2700),("一分半钟",90),("100分钟",6000),("2分30秒",150),("三天",259200)]:
            resolved = resolve_reminder_time(expression+"后提醒我休息", anchor)
            self.assertEqual(resolved.due, anchor + timedelta(seconds=seconds))
        self.assertIsNone(resolve_reminder_time("明天提醒我交报告",anchor).due)
        resolved = resolve_reminder_time("后天晚上七点半提醒我看书",anchor)
        self.assertEqual(resolved.due,datetime(2026,10,15,19,30,tzinfo=anchor.tzinfo))


if __name__ == "__main__": unittest.main()
