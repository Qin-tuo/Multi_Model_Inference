"""Actual SDK guards and mature temporal parsing, with fixture memory vectors."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.temporal import resolve_reminder_time
from app.tools import Args, ToolTurn, CreateReminderArgs, RememberArgs, CreateNoteArgs, ForgetArgs, CancelReminderArgs
from test_mem0_adapter import actual_sdk


class TrustedToolPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_topic_does_not_admit_an_inventory_even_if_model_requests_it(self):
        for text in ('我在研究关于记忆的心理学理论，特别是什么技巧？',
                     '短期记忆有哪些特点？','列出提高记忆的方法。',
                     '我记住了什么英语单词？','请解释“列出我的记忆”是什么意思。'):
            with self.subTest(text=text):
                turn=self.turn(text)
                self.assertIsNone(turn.requested_lookup())
                self.assertEqual(turn.initial_tool_choice(),'auto')
                self.assertEqual(turn.schemas(names={'list_memories'}).standard_tools,[])
                result=await turn.execute('list_memories',Args,{})
                self.assertEqual(result['code'],'EXPLICIT_REQUEST_REQUIRED')

    async def test_explicit_inventory_and_maintenance_keep_real_memory_lookup(self):
        self.memory.remember('owner','fact','fixture','合成已保存事实',source_id='source',request_id='source')
        for text in ('列出我的记忆。','查询已经记住了什么。','你还记住了哪些信息？',
                     '查看所有已保存的记忆。','我的记忆列表有哪些？','请告诉我你记住了什么。'):
            with self.subTest(text=text):
                turn=self.turn(text)
                self.assertEqual(turn.requested_lookup(),'list_memories')
                self.assertEqual(turn.initial_tool_choice(),'required')
                result=await turn.execute('list_memories',Args,{})
                self.assertTrue(result['ok'])
                self.assertEqual(result['result']['items'][0]['content'],'合成已保存事实')
        for text in ('请忘记合成已保存事实。','更正合成已保存事实为新事实。'):
            result=await self.turn(text).execute('list_memories',Args,{})
            self.assertTrue(result['ok'])

    async def test_successful_explicit_inventory_cannot_loop_in_model_continuation(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.utils.types import NOT_GIVEN
        turn=self.turn('列出我的记忆。')
        schema=turn.schemas()
        context=LLMContext(messages=[],tools=schema,tool_choice='required')
        callback=AsyncMock()
        params=SimpleNamespace(arguments={},context=context,tool_call_id='inventory',result_callback=callback)
        await schema.standard_tools[0].handler(params)
        self.assertTrue(turn.calls[0]['outcome']['ok'])
        self.assertIs(context.tools,NOT_GIVEN)
        self.assertEqual(context.tool_choice,'auto')
        callback.assert_awaited_once()

    async def test_read_listing_heading_preserves_data_without_claiming_new_write(self):
        self.records.create_note('owner',title='会议',body='周四讨论接口',source_id='fixture',request_id='fixture')
        turn=self.turn('列出已经保存的笔记。')
        self.assertTrue((await turn.execute('list_notes',Args,{}))['ok'])
        body='\n- 标题：会议\n- 内容：周四讨论接口'
        expected='当前笔记：\n1. 会议：周四讨论接口'
        self.assertEqual(turn.safe_response('我已经保存了以下笔记：'+body),expected)
        self.assertEqual(turn.safe_response('已保存的笔记如下：'+body),expected)
        self.assertEqual(turn.safe_response('周四讨论接口。'),expected)
        self.assertEqual(turn.safe_response('我已经删除了这条笔记。'),expected)
        self.assertEqual(self.turn('你好').safe_response('我已经保存了以下笔记：'+body),'明白了。')

    async def test_maintenance_lookup_keeps_followup_cancel_even_with_listing_words(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from pipecat.processors.aggregators.llm_context import LLMContext
        self.reminders.create('owner','喝水',datetime.now(timezone.utc)+timedelta(days=1),request_id='fixture')
        turn=self.turn('取消刚才列出的提醒。')
        self.assertEqual(turn.required_operation(),'cancel_reminder')
        self.assertEqual(turn.requested_lookup(),'list_reminders')
        schemas=turn.schemas();context=LLMContext(messages=[],tools=schemas,tool_choice='required')
        handler=next(s.handler for s in schemas.standard_tools if s.name=='list_reminders')
        await handler(SimpleNamespace(arguments={},context=context,tool_call_id='lookup',result_callback=AsyncMock()))
        self.assertIn('cancel_reminder',[s.name for s in context.tools.standard_tools])

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
