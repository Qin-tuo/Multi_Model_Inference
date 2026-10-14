"""Payload bounds, epoch changes, and real Pipecat request ownership."""
import asyncio
import threading
import unittest
from unittest.mock import patch

from pipecat.frames.frames import EndFrame, InterruptionFrame, UserImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner
from app.vision_frames import RGBDFrame, VisionFrameStore, VisionUnavailable
from app.vision_snapshot import VisionSnapshotProcessor


class Clock:
    def __init__(self): self.now = 10_000_000_000
    def __call__(self): return self.now


def add(store, clock, *, source=None, age=0, rgb=None, **changes):
    values = dict(source_time_ns=source or clock.now, capture_time_ns=clock.now-age,
        arrival_time_ns=clock.now, width=2, height=2, rgb_step=6, rgb_encoding="rgb8",
        rgb=rgb or bytes(range(12)), depth_step=4, depth=bytes(8), optical_frame="fixture_optical")
    values.update(changes)
    return store.add(**values)


class VisionCacheTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = VisionFrameStore(clock=self.clock, max_bytes=40, max_frame_bytes=30)

    def test_capacity_counts_retired_but_leased_payloads(self):
        add(self.store, self.clock)
        lease = self.store.acquire()
        original = lease.frame.frame_id
        for _ in range(8):
            self.clock.now += 10_000_000
            self.assertTrue(add(self.store, self.clock))
            self.assertLessEqual(self.store.status()["resident_payload_bytes"], 40)
        self.assertIn(original, self.store.resident)
        self.assertNotIn(original, self.store.queue)
        self.assertEqual(self.store.status()["leased_frames"], 1)
        with self.assertRaisesRegex(VisionUnavailable, "BUSY"): self.store.acquire()
        lease.close(); lease.close()
        self.assertEqual(self.store.status()["resident_payload_bytes"], 20)

    def test_age_and_question_skew_are_enforced(self):
        add(self.store, self.clock)
        with self.assertRaisesRegex(VisionUnavailable, "NEAR_QUESTION"):
            self.store.acquire(self.clock.now - 400_000_000)
        self.clock.now += 600_000_000
        self.assertEqual(self.store.status()["state"], "STALE")
        with self.assertRaisesRegex(VisionUnavailable, "STALE"): self.store.acquire()
        self.clock.now += 2_000_000_000
        self.assertEqual(self.store.status()["resident_payload_bytes"], 0)

    def test_epoch_reset_invalidates_lease_without_freeing_live_data(self):
        add(self.store, self.clock)
        lease = self.store.acquire()
        self.store.reset("fixture_disconnect")
        self.assertEqual(self.store.status()["resident_payload_bytes"], 20)
        with self.assertRaisesRegex(VisionUnavailable, "EPOCH"): lease.validate()
        lease.close()
        self.assertEqual(self.store.status()["resident_payload_bytes"], 0)

    def test_invalid_payload_and_source_reset(self):
        self.assertFalse(add(self.store, self.clock, depth_step=9))
        self.assertFalse(add(self.store, self.clock, age=-1))
        self.assertTrue(add(self.store, self.clock))
        self.assertFalse(add(self.store, self.clock))
        before = self.store.epoch
        self.clock.now += 10_000_000
        self.assertTrue(add(self.store, self.clock, source=1_000_000_000))
        self.assertNotEqual(self.store.epoch, before)

    def test_padding_and_bgr_conversion_preserve_pixels(self):
        source = bytes([1, 2, 3, 4, 5, 6, 99, 99] * 2)
        self.assertTrue(add(self.store, self.clock, rgb=source, rgb_step=8, rgb_encoding="bgr8"))
        with self.store.acquire() as lease:
            self.assertEqual(lease.frame.packed_rgb(), bytes([3, 2, 1, 6, 5, 4] * 2))


class Collector(FrameProcessor):
    def __init__(self):
        super().__init__(); self.images = []; self.interruptions = asyncio.Event()
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserImageRawFrame): self.images.append(frame)
        if isinstance(frame, InterruptionFrame): self.interruptions.set()
        await self.push_frame(frame, direction)


class SnapshotPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock = Clock()
        self.store = VisionFrameStore(clock=self.clock)
        add(self.store, self.clock)
        self.provider, self.collector = VisionSnapshotProcessor(self.store), Collector()
        self.worker = PipelineWorker(Pipeline([self.provider, self.collector]), enable_rtvi=False, idle_timeout_secs=None)
        self.errors = []
        @self.worker.event_handler("on_pipeline_error")
        async def error(_worker, frame): self.errors.append(frame.error)
        self.runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        await self.runner.add_workers(self.worker)
        self.run_task = asyncio.create_task(self.runner.run())

    async def asyncTearDown(self):
        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self.run_task, 5)

    async def wait_for(self, predicate):
        async with asyncio.timeout(3):
            while not predicate(): await asyncio.sleep(.01)

    async def test_native_image_request_preserves_metadata_and_explicit_lease(self):
        request = self.provider.request("fixture image", user_id="owner", turn_id="turn-1", question_time_ns=self.clock.now)
        await self.worker.queue_frame(request)
        await self.wait_for(lambda: bool(self.collector.images))
        image = self.collector.images[0]
        self.assertEqual(image.image, bytes(range(12)))
        self.assertEqual(image.metadata["turn_id"], "turn-1")
        self.assertEqual(image.metadata["capture_time_ns"], self.clock.now)
        self.assertEqual(self.store.status()["leased_frames"], 1)
        self.provider.release_request(request.metadata["request_id"])
        self.assertEqual(self.store.status()["leased_frames"], 0)

    async def test_interruption_is_immediate_but_owner_lives_until_packing_finishes(self):
        entered, release = threading.Event(), threading.Event()
        original = RGBDFrame.packed_rgb
        def delayed(frame):
            entered.set()
            if not release.wait(3): raise RuntimeError("Fixture release timed out")
            return original(frame)
        with patch.object(RGBDFrame, "packed_rgb", delayed):
            request = self.provider.request("fixture", user_id="owner", turn_id="old", question_time_ns=self.clock.now)
            await self.worker.queue_frame(request)
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                await self.worker.queue_frame(InterruptionFrame())
                await asyncio.wait_for(self.collector.interruptions.wait(), .5)
                self.assertEqual(self.store.status()["leased_frames"], 1)
                with self.assertRaisesRegex(VisionUnavailable, "BUSY"): self.store.acquire()
            finally: release.set()
            await self.wait_for(lambda: not self.provider.packing)
        self.assertFalse(self.collector.images)
        self.assertEqual(self.store.status()["leased_frames"], 0)

    async def test_late_old_request_cannot_fail_or_modify_a_new_turn(self):
        old = self.provider.request("old", user_id="owner", turn_id="old", question_time_ns=self.clock.now)
        await self.worker.queue_frame(InterruptionFrame())
        await asyncio.wait_for(self.collector.interruptions.wait(), .5)
        await self.worker.queue_frame(old)
        await asyncio.sleep(.05)
        self.assertFalse(self.collector.images)
        self.assertFalse(self.errors)
        self.assertEqual(self.store.counts["stale_requests_dropped"], 1)

    async def test_pending_request_cannot_select_a_new_camera_epoch(self):
        request=self.provider.request('before seek',user_id='owner',turn_id='old',question_time_ns=self.clock.now)
        self.store.reset('recording seek')
        add(self.store,self.clock,rgb=bytes([0,0,255])*4)
        await self.worker.queue_frame(request)
        await self.wait_for(lambda:bool(self.collector.images) or bool(self.errors))
        self.assertFalse(self.collector.images)
        self.assertIn('CAMERA_EPOCH_CHANGED',self.errors)


if __name__ == "__main__": unittest.main()
