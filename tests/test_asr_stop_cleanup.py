"""Native Pipecat timers must retain ownership through repeated VAD stops."""
import asyncio
from types import SimpleNamespace
import unittest

from pipecat.frames.frames import STTMetadataFrame,TranscriptionFrame,VADUserStoppedSpeakingFrame
from pipecat.utils.asyncio.task_manager import TaskManager
from app.adapters.asr import ABORT_GENERATION,ASR_SEQUENCE,CompleteASRTurnStop


class StopTimerCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager=TaskManager()
        self.strategy=CompleteASRTurnStop(SimpleNamespace(complete_through=lambda _:False),user_speech_timeout=10)
        await self.strategy.setup(SimpleNamespace(task_manager=self.manager))
        await self.strategy.process_frame(STTMetadataFrame(service_name='FIXTURE',ttfs_p99_latency=30))

    async def asyncTearDown(self):
        await self.strategy.cleanup()
        for task in list(self.manager.current_tasks()):await self.manager.cancel_task(task)

    def live(self):return [task for task in self.manager.current_tasks() if not task.done()]

    async def test_duplicate_stop_keeps_one_timer_of_each_kind_and_cleanup_joins_both(self):
        for _ in range(3):
            await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=.4))
            await asyncio.sleep(0)
            self.assertEqual(len(self.live()),2)
        await self.strategy.cleanup()
        self.assertEqual(self.live(),[])

    async def test_abort_reset_clears_waits_without_arming_new_timers(self):
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=.4))
        frame=VADUserStoppedSpeakingFrame();frame.metadata[ABORT_GENERATION]=7
        await self.strategy.process_frame(frame)
        self.assertEqual(self.live(),[])
        self.assertFalse(self.strategy._vad_user_speaking)
        self.assertFalse(self.strategy._vad_stopped)

    async def test_final_transcription_and_repeated_stops_leave_no_old_stt_wait(self):
        for _ in range(2):await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=.4))
        frame=TranscriptionFrame('完整语音','owner','FIXTURE',finalized=True)
        frame.metadata[ASR_SEQUENCE]=1
        await self.strategy.process_frame(frame)
        self.assertEqual(len(self.live()),1)
        self.assertTrue(all('_user_speech_timeout_handler' in task.get_name() for task in self.live()))
        await self.strategy.cleanup()
        self.assertEqual(self.live(),[])
