"""Framework wiring tests only; these make no claims about model quality."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService

from app.adapters.llama import LlamaConfig
from app.pipecat_bot import build_pipeline, input_messages, run_smoke


class FrameworkFakeLLM(FrameProcessor):
    """A labelled test processor that exercises real Pipecat frame handling."""

    def __init__(self):
        super().__init__()
        self.context_frames_seen = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.context_frames_seen += 1
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame("FRAMEWORK_TEST_ONLY"))
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class PipecatConfigurationTest(unittest.TestCase):
    def test_local_endpoint_and_manifest_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps({"models": {"vlm": {
                "backend": "llama.cpp", "repository": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
                "max_output_tokens": 128,
            }}}))
            config = LlamaConfig.from_manifest(path)
            self.assertEqual(config.model, "Qwen2.5-VL-3B-Instruct")
            self.assertEqual(config.max_tokens, 128)
        with self.assertRaises(ValueError):
            LlamaConfig("model", "https://api.openai.com/v1")

    def test_bounded_file_inputs(self):
        with self.assertRaises(ValueError):
            input_messages(" ")
        with self.assertRaises(ValueError):
            input_messages("a" * 65537)
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "tiny.png"
            image.write_bytes(b"test-file-only")
            messages = input_messages("describe", image)
            self.assertEqual(messages[-1]["content"][0]["type"], "text")
            self.assertTrue(messages[-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))


class PipecatPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_maintained_framework_objects(self):
        bot = build_pipeline(LlamaConfig("manifest-model"), input_messages("test"))
        self.assertIsInstance(bot.worker, PipelineWorker)
        self.assertIsInstance(bot.llm, OpenAILLMService)
        self.assertIn(bot.aggregators.user(), bot.pipeline.processors)
        self.assertIn(bot.aggregators.assistant(), bot.pipeline.processors)

    async def test_native_runner_context_and_completion_path(self):
        fake = FrameworkFakeLLM()
        result = await run_smoke(
            LlamaConfig("framework-test-model"), input_messages("test"),
            llm=fake, timeout_seconds=10,
        )
        self.assertEqual(fake.context_frames_seen, 1)
        self.assertEqual(result["response"], "FRAMEWORK_TEST_ONLY")
        self.assertFalse(result["real_model_execution"])
        self.assertEqual(result["backend"], "injected_framework_test_processor")
        self.assertGreaterEqual(result["context_message_count"], 3)


if __name__ == "__main__":
    unittest.main()
