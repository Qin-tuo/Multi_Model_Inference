"""Pipecat/OVS PCM and failure contracts; mock HTTP is not speech quality evidence."""
import asyncio
import json
import unittest

import httpx
from pipecat.frames.frames import EndFrame, OutputAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from app.adapters.speech import OVSTTSService


class AudioCapture(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame):
            self.frames.append((frame.audio,frame.sample_rate,frame.num_channels))
        await self.push_frame(frame,direction)


class SpeechTransportTests(unittest.IsolatedAsyncioTestCase):
    async def run_transport(self, response):
        calls = []
        def handle(request):
            calls.append(json.loads(request.content))
            return response
        service = OVSTTSService()
        await service._http.aclose()
        service._http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        capture = AudioCapture()
        worker = PipelineWorker(Pipeline([service,capture]), params=PipelineParams(audio_out_sample_rate=16000),
                                enable_rtvi=False,idle_timeout_secs=None)
        errors = []
        @worker.event_handler("on_pipeline_error")
        async def error(_worker, frame):
            errors.append(frame.error)
        runner = WorkerRunner(handle_sigint=False,handle_sigterm=False)
        await runner.add_workers(worker)
        await worker.queue_frames([TTSSpeakFrame("传输合同测试"),EndFrame()])
        async with asyncio.timeout(10):
            await runner.run()
        self.assertTrue(service._http.is_closed)
        return calls,capture.frames,errors

    async def test_pcm16_rate_voice_and_complete_payload(self):
        pcm = bytes(range(256))*5
        calls,frames,errors = await self.run_transport(httpx.Response(200,content=pcm))
        self.assertEqual(calls[0]["voice"],"0")
        self.assertEqual(calls[0]["response_format"],"pcm")
        self.assertEqual(b"".join(f[0] for f in frames),pcm)
        self.assertTrue(all(f[1:]==(16000,1) for f in frames))
        self.assertFalse(errors)

    async def test_http_failure_has_no_audio_success(self):
        _,frames,errors = await self.run_transport(httpx.Response(503,text="unavailable"))
        self.assertFalse(frames)
        self.assertTrue(errors)

    async def test_partial_sample_is_rejected(self):
        _,frames,errors = await self.run_transport(httpx.Response(200,content=b"\x00"))
        self.assertFalse(frames)
        self.assertTrue(errors)


if __name__=="__main__":
    unittest.main()
