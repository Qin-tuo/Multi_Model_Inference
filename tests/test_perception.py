"""Labelled numeric pose trajectories and real SQLite policy; no camera accuracy claim."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from sqlalchemy import select
from app.perception import GestureRules, Person
from app.proactive import VisualFeedback
from app.reminders import ReminderService, _NOTIFICATIONS


class PoseFixture:
    def __init__(self):
        self.now = 1_000_000_000
        self.rules = GestureRules(clock=lambda:self.now)
    def person(self, *, raised=True, x=0, track='1', missing=False):
        points={6:(280,240,.9),8:(280,205,.9),10:(280+x*200,170 if raised else 330,.9),
                7:(360,240,.9),9:(370,290,.9),11:(380,330,.9)}
        if missing: points[10]=(280,170,.1)
        return Person(track,.9,320,240,200,360,points,1.5)
    def feed(self, people=None, *, source='fixture-publisher', steps=1):
        for _ in range(steps):
            self.now += 100_000_000
            self.rules.update([self.person()] if people is None else people,source_key=source,
                source_stamp_ns=100_000_000_000+self.now,capture_ns=self.now)
    def raised_event(self):
        self.feed(steps=14)
        return self.rules.drain_events()[0]


class GestureTest(unittest.TestCase):
    def setUp(self): self.f=PoseFixture()
    def test_static_raise_deduplicates_then_neutral_rearms(self):
        event=self.f.raised_event()
        self.assertEqual(event['kind'],'hand_raise')
        self.f.feed(steps=20)
        self.assertFalse(self.f.rules.drain_events())
        self.f.feed([self.f.person(raised=False)],steps=7)
        release=self.f.rules.snapshot()['people'][0]['released_at_ns']
        self.f.feed([self.f.person(raised=False)],steps=3)
        self.assertEqual(release,self.f.rules.snapshot()['people'][0]['released_at_ns'])
        self.f.feed(steps=14)
        self.assertEqual(len(self.f.rules.drain_events()),1)
    def test_wave_requires_reversals_and_uses_one_based_ids(self):
        for x in [0,.3,-.3,.3,-.3]: self.f.feed([self.f.person(x=x)])
        self.assertEqual(self.f.rules.drain_events()[0]['kind'],'wave')
        self.f.feed(steps=20)
        self.assertFalse(self.f.rules.drain_events())
    def test_noise_missing_joints_and_short_holds_do_not_trigger(self):
        self.f.feed([self.f.person(missing=True)],steps=30)
        self.f.feed(steps=7)
        self.f.feed([self.f.person(raised=False)])
        self.f.feed(steps=7)
        self.assertFalse(self.f.rules.drain_events())
        for _ in range(30): self.f.feed([self.f.person(raised=False,x=.01)])
        self.assertFalse(self.f.rules.drain_events())
    def test_preserving_weak_coordinates_keeps_gesture_decisions(self):
        for confidence in (.1,.49,.59,.6,.9):
            with self.subTest(confidence=confidence):
                full,sparse=PoseFixture(),PoseFixture()
                for x in [0,.3,-.3,.3,-.3]+[0]*20:
                    person=full.person(x=x)
                    points={i:(320,240,.1) for i in range(1,18)}
                    points.update(person.keypoints)
                    px,py,_=points[10];points[10]=(px,py,confidence)
                    full.feed([replace(person,keypoints=points)])
                    sparse.feed([replace(person,keypoints={i:p for i,p in points.items() if p[2]>=.5})])
                events=lambda f:[(e['kind'],e['event_id'],e['capture_ns']) for e in f.rules.drain_events()]
                self.assertEqual(events(full),events(sparse))
                self.assertEqual(full.rules.counts['invalid_person'],0)
    def test_occlusion_does_not_count_as_hold_or_rearm_emitted_pose(self):
        self.f.feed(steps=7)
        self.f.feed([],steps=7)
        self.f.feed(steps=7)
        self.assertFalse(self.f.rules.drain_events())
        self.f.feed(steps=7)
        self.assertEqual(len(self.f.rules.drain_events()),1)
        self.f.feed([],steps=7)
        self.f.feed(steps=20)
        self.assertFalse(self.f.rules.drain_events())
    def test_deadline_source_restart_and_stream_gap_revoke(self):
        event=self.f.raised_event()
        self.assertTrue(self.f.rules.valid(event))
        self.f.feed(steps=21)
        self.assertFalse(self.f.rules.valid(event))
        self.assertTrue(self.f.rules.valid(event,started=True))
        self.f.feed(source='new-publisher')
        self.assertFalse(self.f.rules.valid(event,started=True))
        self.f.now+=600_000_000
        self.assertEqual(self.f.rules.snapshot()['state'],'STALE')
    def test_duplicate_frames_invalid_data_and_bounds(self):
        self.f.feed()
        self.f.rules.update([self.f.person()],source_key='fixture-publisher',
            source_stamp_ns=100_000_000_000+self.f.now,capture_ns=self.f.now)
        self.assertEqual(self.f.rules.counts['duplicate_or_out_of_order'],1)
        self.f.feed([replace(self.f.person(),width=float('nan'))],steps=20)
        self.assertFalse(self.f.rules.drain_events())
        self.f.feed([self.f.person(track=str(i)) for i in range(40)],steps=14)
        self.assertLessEqual(len(self.f.rules.tracks),16)
        self.assertLessEqual(len(self.f.rules.events),32)
        with self.assertRaises(ValueError): GestureRules(stale_seconds=-1)


class FeedbackTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db=Path(self.temp.name)/'reminders.sqlite'
        self.f=PoseFixture()
        self.open()
    def open(self):
        self.service=ReminderService(self.db)
        self.feedback=VisualFeedback(self.service,self.f.rules,'owner',clock=lambda:self.f.now)
        self.addCleanup(self.service.close)
        self.addCleanup(self.feedback.close)
    def trigger(self):
        self.f.feed(steps=14); self.feedback.tick()
        return self.service.claim_next('owner')
    def states(self):
        with self.service.engine.connect() as c: return [dict(r) for r in c.execute(select(_NOTIFICATIONS)).mappings()]
    def test_greeting_reuses_outbox_and_is_not_a_user_reminder(self):
        notice=self.trigger()
        self.assertEqual(notice['text'],'你好，有什么可以帮你？')
        self.assertEqual(self.service.list_reminders('owner',include_inactive=True),[])
        self.assertIsNone(self.service.claim_next('guest'))
        self.assertTrue(self.service.begin_output(notice['id'],attempt_id=notice['attempt_id'],channel='fixture'))
        self.assertTrue(self.service.acknowledge_delivery(notice['id'],attempt_id=notice['attempt_id'],evidence='labelled fixture drain'))
        self.f.feed(steps=20); self.feedback.tick()
        self.assertIsNone(self.service.claim_next('owner'))
        self.assertEqual(self.states()[0]['status'],'fixture_delivered')
        self.assertIsNone(self.states()[0]['played_utc'])
    def test_foreground_delay_expiry_and_no_late_greeting(self):
        self.f.feed(steps=14); self.feedback.tick()
        self.assertIsNone(self.service.claim_next('owner',occupied=True))
        self.f.feed(steps=21); self.feedback.tick()
        self.assertIsNone(self.service.claim_next('owner'))
        self.assertEqual(self.states()[0]['status'],'expired')
    def test_source_invalidates_claim_and_started_audio(self):
        notice=self.trigger()
        self.assertTrue(self.service.begin_output(notice['id'],attempt_id=notice['attempt_id'],channel='fixture'))
        self.f.feed(source='new-publisher')
        self.assertFalse(self.service.attempt_active(notice))
        self.assertFalse(self.service.acknowledge_delivery(notice['id'],attempt_id=notice['attempt_id'],evidence='stale drain'))
        self.service.fail_output(notice['id'],attempt_id=notice['attempt_id'],reason='source invalid')
        self.assertEqual(self.states()[0]['status'],'uncertain')
    def test_started_output_can_finish_after_deadline_but_cancel_revokes_it(self):
        notice=self.trigger()
        self.assertTrue(self.service.begin_output(notice['id'],attempt_id=notice['attempt_id'],channel='fixture'))
        self.f.feed(steps=21)
        self.assertTrue(self.service.notification_allowed(notice['id']))
        self.service.cancel('owner',notice['reminder_id'],request_id='cancel-fixture-gesture')
        self.assertFalse(self.service.attempt_active(notice))
        self.assertFalse(self.service.notification_allowed(notice['id']))
        self.assertNotIn(notice['attempt_id'],self.service._started_attempts)
        self.assertFalse(self.service.acknowledge_delivery(notice['id'],attempt_id=notice['attempt_id'],evidence='cancelled drain'))
    def test_mute_revokes_audio_and_unmute_never_replays_old_event(self):
        notice=self.trigger()
        self.assertTrue(self.service.attempt_active(notice))
        self.feedback.set_muted(True)
        self.assertFalse(self.service.attempt_active(notice))
        self.feedback.set_muted(False); self.feedback.tick()
        self.assertIsNone(self.service.claim_next('owner'))
        self.assertFalse(self.service.begin_output(notice['id'],attempt_id=notice['attempt_id'],channel='text'))
        self.feedback.set_muted(True)
        self.feedback.close(); self.service.close(); self.open()
        self.assertTrue(self.feedback.muted)
    def test_restart_discards_old_event_and_retains_held_pose_latch(self):
        self.trigger()
        self.feedback.close(); self.service.close()
        self.f.rules=GestureRules(clock=lambda:self.f.now)
        self.open()
        self.assertIsNone(self.service.claim_next('owner'))
        self.assertEqual(self.states()[0]['status'],'expired')
        self.assertIsNone(self.trigger())
        self.assertEqual(self.feedback.counts['latched'],1)
        self.f.feed([self.f.person(raised=False)],steps=8); self.feedback.tick()
        self.assertIsNotNone(self.trigger())
    def test_publisher_change_retains_profile_cooldown(self):
        self.trigger()
        self.f.feed(source='replacement',steps=14); self.feedback.tick()
        self.assertEqual(self.feedback.counts['suppressed'],1)
        self.assertEqual(len(self.states()),1)
    def test_same_profile_has_bounded_pending_work(self):
        self.service.max_pending=1
        self.trigger()
        self.f.feed([self.f.person(),self.f.person(track='2')],steps=14)
        self.feedback.tick()
        self.assertEqual(len(self.states()),1)
