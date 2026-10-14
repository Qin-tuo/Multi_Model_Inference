"""Replay clock admission/epoch policy without substituting ROS integration."""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from app.adapters.vision import RosVisionBridge
from app.vision_frames import VisionUnavailable


class ReplayClockTests(unittest.TestCase):
    def test_simulation_requires_explicit_isolated_domain(self):
        with patch.dict('os.environ',{'ASSISTANT_ROS_DOMAIN_ID':'42'}):
            for domain in (None,42,True):
                with self.subTest(domain=domain),self.assertRaises(ValueError):
                    RosVisionBridge(domain_id=domain,use_sim_time=True)
            with self.assertRaises(ValueError):RosVisionBridge(domain_id=192,use_sim_time='true')
            self.assertFalse(RosVisionBridge().use_sim_time)

    def test_clock_seek_retires_frames_and_invalidates_existing_lease(self):
        bridge=RosVisionBridge(domain_id=192,use_sim_time=True)
        now=10_000_000_000;bridge.frames.clock=lambda:now
        bridge._observe_replay_clock(1_000_000_000,False,now)
        bridge.frames.add(source_time_ns=1_600_000_000_000_000_000,capture_time_ns=now,
            arrival_time_ns=now,width=2,height=2,rgb_step=6,rgb_encoding='rgb8',rgb=bytes(12),
            depth_step=4,depth=bytes(8),optical_frame='fixture')
        lease=bridge.frames.acquire();epoch=bridge.frames.epoch
        bridge.sync=SimpleNamespace(lock=threading.Lock(),queues=[{1:'oldcolor'},{2:'olddepth'}])
        # A normal200ms tick is not a clock discontinuity.
        bridge._observe_replay_clock(200_000_000,False,now+200_000_000)
        self.assertEqual(bridge.frames.epoch,epoch)
        bridge._observe_replay_clock(-2_000_000_000,False,now+210_000_000)
        self.assertTrue(all(not q for q in bridge.sync.queues))
        with self.assertRaisesRegex(VisionUnavailable,'EPOCH'):lease.validate()
        self.assertEqual(bridge.frames.status()['resident_payload_bytes'],20)
        lease.close();self.assertEqual(bridge.frames.status()['resident_payload_bytes'],0)

    def test_frozen_sim_clock_cannot_make_late_input_fresh(self):
        bridge=RosVisionBridge(domain_id=192,use_sim_time=True)
        clock=SimpleNamespace(ros_time_is_active=True,now=lambda:SimpleNamespace(nanoseconds=1_600_000_000_000_000_000))
        bridge.node=SimpleNamespace(get_clock=lambda:clock)
        now=10_000_000_000
        self.assertIsNone(bridge._clock_offset_at(now))
        bridge.last_clock_progress_ns=now
        self.assertIsNotNone(bridge._clock_offset_at(now+100_000_000))
        self.assertIsNone(bridge._clock_offset_at(now+600_000_000))

    def test_late_cached_frame_is_unavailable_when_source_clock_stops(self):
        bridge=RosVisionBridge(domain_id=192,use_sim_time=True)
        now=10_490_000_000;bridge.frames.clock=lambda:now
        bridge.last_clock_progress_ns=10_000_000_000
        bridge.frames.add(source_time_ns=1_600_000_000_000_000_000,capture_time_ns=now,
            arrival_time_ns=now,width=2,height=2,rgb_step=6,rgb_encoding='rgb8',rgb=bytes(12),
            depth_step=4,depth=bytes(8),optical_frame='fixture')
        lease=bridge.frames.acquire()
        now=10_600_000_000
        self.assertEqual(bridge.frames.status()['state'],'STALE')
        with self.assertRaisesRegex(VisionUnavailable,'CLOCK'):lease.validate()
        lease.close()
        with self.assertRaisesRegex(VisionUnavailable,'CLOCK'):bridge.frames.acquire()


if __name__=='__main__':unittest.main()
