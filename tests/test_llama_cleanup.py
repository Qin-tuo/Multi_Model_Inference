"""Native worker shutdown closes the local service's owned SDK connection pool."""
import asyncio
import unittest
from unittest.mock import AsyncMock,patch

from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.workers.runner import WorkerRunner
from app.adapters.llama import LlamaConfig,create_llama_service,provider_slots_idle


class LlamaClientCleanupTests(unittest.IsolatedAsyncioTestCase):
    def test_provider_observation_requires_real_boolean_slot_states(self):
        for value in ([],{},None,[{}],[False],[{'is_processing':None}],[{'is_processing':0}],[{'is_processing':'false'}]):
            with self.subTest(value=value),self.assertRaises(RuntimeError):provider_slots_idle(value)
        self.assertTrue(provider_slots_idle([{'is_processing':False}]))
        self.assertFalse(provider_slots_idle([{'is_processing':False},{'is_processing':True}]))

    async def test_native_end_and_cancel_close_the_client(self):
        for mode in ('end','cancel'):
            with self.subTest(mode=mode):
                service=create_llama_service(LlamaConfig('FIXTURE'))
                worker=PipelineWorker(Pipeline([service]),idle_timeout_secs=None,enable_rtvi=False)
                runner=WorkerRunner(handle_sigint=False,handle_sigterm=False);ready=asyncio.Event()
                @worker.event_handler('on_pipeline_started')
                async def started(_worker,_frame):ready.set()
                await runner.add_workers(worker);running=asyncio.create_task(runner.run())
                try:
                    await asyncio.wait_for(ready.wait(),5)
                    self.assertFalse(service._client.is_closed())
                    if mode=='end':await worker.queue_frame(EndFrame())
                    else:await worker.cancel()
                    await asyncio.wait_for(running,5)
                    self.assertTrue(service._client.is_closed())
                    self.assertEqual([t.get_name() for t in runner.task_manager.current_tasks() if not t.done()],[])
                finally:
                    if not running.done():
                        await runner.cancel();await asyncio.wait_for(running,5)
                    await service._client.close()

    async def test_parent_cleanup_failure_still_releases_owned_client(self):
        service=create_llama_service(LlamaConfig('FIXTURE'))
        try:
            with patch.object(OpenAILLMService,'cleanup',AsyncMock(side_effect=RuntimeError('injected teardown failure'))):
                with self.assertRaisesRegex(RuntimeError,'injected teardown failure'):await service.cleanup()
            self.assertTrue(service._client.is_closed())
        finally:await service._client.close()
