"""Failed startup must release constructor-owned clients without closing stores."""
import asyncio
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock,patch

import httpx
from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.voice_assistant import VoiceAssistant
from test_mem0_adapter import actual_sdk


class VoiceStartupCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.stack=ExitStack();path=Path(self.temp.name)
        self.memory=self.stack.enter_context(Mem0Adapter(path/'memory',sdk=actual_sdk(path/'memory'),embedding_scope='FIXTURE'))
        self.records=self.stack.enter_context(Records(path/'records.sqlite'))
        self.reminders=self.stack.enter_context(ReminderService(path/'reminders.sqlite'))
        self.voice=VoiceAssistant(LlamaConfig('FIXTURE'),self.memory,self.reminders,self.records,output_path=path/'output.wav')

    async def asyncTearDown(self):
        await self.voice.close()
        # Test cleanup cannot make the assertions below pass retroactively.
        await self.voice.stt._client.close();await self.voice.tts._http.aclose()
        await self.voice.session.bot.llm._client.close();await self.voice.session.budget.client.aclose()
        self.stack.close();self.temp.cleanup()

    def assert_released(self):
        self.assertTrue(self.voice.closed)
        self.assertTrue(self.voice.stt._client.is_closed())
        self.assertTrue(self.voice.tts._http.is_closed)
        self.assertTrue(self.voice.session.bot.llm._client.is_closed())
        self.assertTrue(self.voice.session.budget.client.is_closed)
        self.assertEqual(self.memory.list_memories('owner'),[])
        self.assertEqual(self.records.recent('owner'),[])
        self.assertEqual(self.reminders.list_reminders('owner'),[])

    async def test_failed_speech_preflight_closes_unstarted_pipelines(self):
        with patch('app.voice_assistant.validate_speech_capabilities',AsyncMock(side_effect=ValueError('not ready'))):
            with self.assertRaisesRegex(ValueError,'not ready'):await self.voice.start()
        self.assert_released()
        await self.voice.close();self.assert_released()

    async def test_failed_model_health_closes_both_unstarted_pipelines(self):
        client=self.voice.session.budget.client;await client.aclose()
        self.voice.session.budget.client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(503)))
        with self.assertRaises(httpx.HTTPStatusError):await self.voice.start(check_services=False)
        self.assert_released()

    async def test_cancelled_preflight_releases_clients_and_preserves_borrowed_stores(self):
        with patch('app.voice_assistant.validate_speech_capabilities',AsyncMock(side_effect=asyncio.CancelledError())):
            with self.assertRaises(asyncio.CancelledError):await self.voice.start()
        self.assert_released()
