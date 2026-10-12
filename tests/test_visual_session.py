"""Native Pipecat/real stores; image, model and token counts are labelled fixtures."""
import asyncio
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import LLMContextFrame
from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.session import AssistantSession
from app.vision_frames import VisionFrameStore, VisionUnavailable
from app.vision_snapshot import wants_visual_input
from test_mem0_adapter import actual_sdk
from test_pipecat_pipeline import FrameworkFakeLLM


class ImageInspectLLM(FrameworkFakeLLM):
    def __init__(self):
        super().__init__()
        self.contexts = []
        self.on_context = None

    async def process_frame(self, frame, direction):
        if isinstance(frame, LLMContextFrame):
            self.contexts.append(copy.deepcopy(frame.context.get_messages()))
            if self.on_context: self.on_context()
        await super().process_frame(frame, direction)


class VisualSessionTests(unittest.IsolatedAsyncioTestCase):
    def test_spoken_deictic_question_uses_visual_input_without_capturing_model_questions(self):
        self.assertTrue(wants_visual_input("这是什么？"))
        self.assertTrue(wants_visual_input("请问那个是什么？"))
        self.assertFalse(wants_visual_input("这是什么模型？"))

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.memory = Mem0Adapter(root / "memory", sdk=actual_sdk(root / "memory"), embedding_scope="FIXTURE")
        self.records = Records(root / "records.sqlite")
        self.reminders = ReminderService(root / "reminders.sqlite")
        self.frames = VisionFrameStore()
        self.llm = ImageInspectLLM()
        self.session = AssistantSession(LlamaConfig("fixture"), self.memory, self.reminders, self.records,
            llm=self.llm, vision_frames=self.frames)
        self.session.budget.count = AsyncMock(return_value=100)
        self.evidence = patch("app.session.memory_message", return_value=({"role": "user", "content": "FIXTURE []"},
            {"injected_memory_ids": [], "memory_tokens": 3, "retrieved_count": 0, "injected_count": 0}))
        self.evidence.start()
        self.emitted = []
        await self.session.start()

    def add_frame(self):
        now = time.monotonic_ns()
        self.assertTrue(self.frames.add(source_time_ns=now, capture_time_ns=now, arrival_time_ns=now,
            width=2, height=2, rgb_step=6, rgb_encoding="rgb8", rgb=bytes([255, 0, 0]) * 4,
            depth_step=4, depth=bytes(8), optical_frame="FIXTURE"))
        return now

    async def emit(self, text): self.emitted.append(text)

    async def asyncTearDown(self):
        await self.session.close()
        self.evidence.stop()
        self.reminders.close(); self.records.close(); self.memory.close(); self.temp.cleanup()

    async def test_native_image_used_once_and_never_persisted_or_kept_in_context(self):
        capture = self.add_frame()
        result = await self.session.ask("你现在看到了什么？", emit=self.emit, question_time_ns=capture)
        self.assertEqual(result["image"]["question_frame_skew_ms"], 0)
        self.assertIn("image_url", str(self.llm.contexts[0]))
        self.assertNotIn("image_url", str(self.session.bot.context.get_messages()))
        self.assertEqual(self.frames.status()["leased_frames"], 0)
        self.assertNotIn("data:image", str(self.records.recent("owner")))
        await self.session.ask("一加一等于几？", emit=self.emit)
        self.assertNotIn("image_url", str(self.llm.contexts[1]))

    async def test_no_frame_refuses_before_model_and_does_not_save_an_answer(self):
        with self.assertRaisesRegex(RuntimeError, "NO_FRESH_FRAME"):
            await self.session.ask("镜头里是什么？", emit=self.emit)
        self.assertEqual(self.llm.context_frames_seen, 0)
        self.assertEqual(self.records.recent("owner"), [])
        self.assertFalse(self.emitted)

    async def test_camera_epoch_change_during_budgeting_prevents_model_admission(self):
        self.add_frame()
        async def reset(_context):
            self.frames.reset("FIXTURE_RESTART")
            return 100
        self.session.budget.count.side_effect = reset
        with self.assertRaisesRegex(RuntimeError, "CAMERA_EPOCH_CHANGED"):
            await self.session.ask("画面是什么？", emit=self.emit)
        self.assertEqual(self.llm.context_frames_seen, 0)
        self.assertEqual(self.frames.status()["leased_frames"], 0)

    async def test_camera_epoch_change_during_generation_discards_the_answer(self):
        self.add_frame()
        self.llm.on_context = lambda: self.frames.reset("FIXTURE_RESTART")
        with self.assertRaisesRegex(VisionUnavailable, "CAMERA_EPOCH_CHANGED"):
            await self.session.ask("画面是什么？", emit=self.emit)
        self.assertEqual(self.llm.context_frames_seen, 1)
        self.assertFalse(self.emitted)
        self.assertEqual(self.records.recent("owner"), [])
        self.assertEqual(self.frames.status()["leased_frames"], 0)

    async def test_cancel_during_preparation_cannot_admit_a_later_model_request(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.session._messages
        async def delayed(*args):
            entered.set()
            await release.wait()
            return await original(*args)
        with patch.object(self.session, "_messages", delayed):
            pending = asyncio.create_task(self.session.ask("一加一", emit=self.emit))
            await asyncio.wait_for(entered.wait(), 2)
            await self.session.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError): await pending
        self.assertEqual(self.llm.context_frames_seen, 0)
        self.assertFalse(self.emitted)
        await self.session.ask("一加二", emit=self.emit)
        self.assertEqual(self.llm.context_frames_seen, 1)

    async def test_oversized_multimodal_prompt_releases_image_without_inference(self):
        self.add_frame()
        self.session.budget.count.return_value = 2000
        with self.assertRaises(ValueError):
            await self.session.ask("看一下", emit=self.emit, visual=True)
        self.assertEqual(self.llm.context_frames_seen, 0)
        self.assertEqual(self.frames.status()["leased_frames"], 0)


if __name__ == "__main__": unittest.main()
