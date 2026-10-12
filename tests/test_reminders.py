"""Real APScheduler/SQLite integration; playback is deliberately not simulated."""

from __future__ import annotations

import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from app.reminders import ReminderService, _NOTIFICATIONS, _deliver_due


class ReminderIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "reminders.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def wait_fired(self, service: ReminderService, reminder_id: str):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            rows = service.list_reminders("owner", include_inactive=True)
            for row in rows:
                if row["id"] == reminder_id and row["status"] == "fired":
                    return
            time.sleep(0.01)
        self.fail("real APScheduler did not produce the due notification")

    @staticmethod
    def due_after(seconds: float):
        return datetime.now(timezone.utc) + timedelta(seconds=seconds)

    def test_duplicate_create_capacity_and_single_owner(self):
        with ReminderService(self.database, max_pending=1) as service:
            due = self.due_after(60)
            first = service.create("owner", "same reminder", due, request_id="create-1")
            repeated = service.create("owner", "same reminder", due, request_id="create-1")
            self.assertEqual(first, repeated)
            self.assertEqual(len(service.scheduler.get_jobs()), 1)
            self.assertEqual(len(service.list_reminders("owner")), 1)
            self.assertEqual(service.list_reminders("guest"), [])
            with self.assertRaises(ValueError):
                service.create("owner", "different", due, request_id="create-1")
            with self.assertRaises(RuntimeError):
                service.create("owner", "too many", due, request_id="create-2")
            with self.assertRaises(RuntimeError):
                ReminderService(self.database)

    def test_cancel_before_due_and_invalidate_already_claimed_notice(self):
        with ReminderService(self.database) as service:
            early = service.create("owner", "cancel before due", self.due_after(0.15), request_id="early")
            cancelled = service.cancel("owner", early["reminder_id"], request_id="cancel-early")
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(service.cancel("owner", early["reminder_id"], request_id="cancel-early"), cancelled)
            time.sleep(0.2)
            _deliver_due(str(self.database), early["reminder_id"])  # late persisted callback stays harmless
            self.assertIsNone(service.claim_next("owner"))
            later = service.create("owner", "cancel pending playback", self.due_after(0.05), request_id="later")
            self.wait_fired(service, later["reminder_id"])
            notice = service.claim_next("owner")
            self.assertTrue(service.notification_allowed(notice["id"]))
            service.cancel("owner", later["reminder_id"], request_id="cancel-later")
            self.assertFalse(service.notification_allowed(notice["id"]))
            self.assertFalse(service.acknowledge_delivery(notice["id"], attempt_id=notice["attempt_id"], evidence="fixture"))
            self.assertIsNone(service.claim_next("owner"))

    def test_restart_recovers_missing_job_and_deduplicates_due_callback(self):
        with ReminderService(self.database) as original:
            reminder = original.create("owner", "recover durable reminder", self.due_after(0.1), request_id="durable")
            original.scheduler.remove_job("reminder:" + reminder["reminder_id"])
        time.sleep(0.15)
        with ReminderService(self.database) as restarted:
            self.wait_fired(restarted, reminder["reminder_id"])
            _deliver_due(str(self.database), reminder["reminder_id"])
            with restarted.engine.connect() as connection:
                notices = connection.execute(select(_NOTIFICATIONS)).mappings().all()
                self.assertEqual(len(notices), 1)
            notice = restarted.claim_next("owner")
            notice_id = notice["id"]
        # Crash-equivalent unacknowledged claim recovers with the same logical ID.
        with ReminderService(self.database) as after_claim_restart:
            notice = after_claim_restart.claim_next("owner")
            self.assertEqual(notice["id"], notice_id)
            self.assertEqual(notice["delivery_attempts"], 2)
            self.assertTrue(after_claim_restart.begin_output(notice_id, attempt_id=notice["attempt_id"], channel="fixture"))
            self.assertTrue(after_claim_restart.acknowledge_delivery(notice_id, attempt_id=notice["attempt_id"], evidence="fixture-drained"))
            self.assertTrue(after_claim_restart.acknowledge_delivery(notice_id, attempt_id=notice["attempt_id"], evidence="fixture-drained"))
            self.assertIsNone(after_claim_restart.claim_next("owner"))

    def test_mute_and_occupied_keep_notice_unplayed(self):
        with ReminderService(self.database) as service:
            reminder = service.create("owner", "respect playback gate", self.due_after(0.05), request_id="gate")
            self.wait_fired(service, reminder["reminder_id"])
            self.assertIsNone(service.claim_next("owner", muted=True))
            self.assertIsNone(service.claim_next("owner", occupied=True))
            self.assertIsNone(service.claim_next("guest"))
            notice = service.claim_next("owner")
            self.assertEqual(notice["status"], "claimed")
            self.assertEqual(notice["delivery_attempts"], 1)
            self.assertIsNone(service.claim_next("owner"))
            self.assertTrue(service.release_notification(notice["id"], attempt_id=notice["attempt_id"]))
            retried = service.claim_next("owner")
            self.assertEqual(retried["id"], notice["id"])
            self.assertEqual(retried["delivery_attempts"], 2)
            # Tests acknowledge the delivery protocol, not an acoustic measurement.
            self.assertTrue(service.begin_output(retried["id"], attempt_id=retried["attempt_id"], channel="fixture"))
            self.assertTrue(service.acknowledge_delivery(retried["id"], attempt_id=retried["attempt_id"], evidence="fixture-drained"))
            cancelled = service.cancel("owner", reminder["reminder_id"], request_id="cancel-history")
            self.assertFalse(cancelled["played_output_rolled_back"])
            with service.engine.connect() as connection:
                state = connection.execute(select(_NOTIFICATIONS.c.status)).scalar_one()
                self.assertEqual(state, "fixture_delivered")

    def test_started_unconfirmed_output_is_not_replayed_after_restart(self):
        with ReminderService(self.database) as service:
            result = service.create("owner", "uncertain output", self.due_after(.05), request_id="uncertain")
            self.wait_fired(service, result["reminder_id"])
            notice = service.claim_next("owner")
            self.assertTrue(service.begin_output(notice["id"], attempt_id=notice["attempt_id"], channel="fixture"))
        with ReminderService(self.database) as restarted:
            self.assertIsNone(restarted.claim_next("owner"))
            self.assertEqual(restarted.list_reminders("owner")[0]["status"], "uncertain")
            self.assertFalse(restarted.acknowledge_delivery(notice["id"], attempt_id=notice["attempt_id"], evidence="stale"))

    def test_stale_attempt_and_early_ack_are_rejected(self):
        with ReminderService(self.database) as service:
            result = service.create("owner", "attempt identity", self.due_after(.05), request_id="attempt")
            self.wait_fired(service, result["reminder_id"])
            old = service.claim_next("owner")
            self.assertFalse(service.acknowledge_delivery(old["id"], attempt_id=old["attempt_id"], evidence="synthesized-only"))
            service.release_notification(old["id"], attempt_id=old["attempt_id"])
            current = service.claim_next("owner")
            self.assertNotEqual(old["attempt_id"], current["attempt_id"])
            self.assertFalse(service.begin_output(old["id"], attempt_id=old["attempt_id"], channel="fixture"))
            self.assertTrue(service.begin_output(current["id"], attempt_id=current["attempt_id"], channel="fixture"))
            self.assertFalse(service.acknowledge_delivery(old["id"], attempt_id=old["attempt_id"], evidence="late-receipt"))

    def test_failed_unstarted_output_has_bounded_retries(self):
        with ReminderService(self.database, max_delivery_attempts=2, retry_delay_seconds=0) as service:
            result = service.create("owner", "failed TTS", self.due_after(.05), request_id="failure")
            self.wait_fired(service, result["reminder_id"])
            for _ in range(2):
                notice = service.claim_next("owner")
                self.assertTrue(service.fail_output(notice["id"], attempt_id=notice["attempt_id"], reason="fixture synth failure"))
            self.assertIsNone(service.claim_next("owner"))
            self.assertEqual(service.list_reminders("owner")[0]["status"], "failed")

    def test_expired_misfire_has_no_output_and_pending_notice_expires(self):
        with ReminderService(self.database, notification_ttl_seconds=1) as service:
            old = service.create("owner", "already expired", self.due_after(-3), request_id="expired")
            _deliver_due(str(self.database), old["reminder_id"])
            self.assertIsNone(service.claim_next("owner"))
            pending = service.create("owner", "expires while muted", self.due_after(.05), request_id="pending-expiry")
            self.wait_fired(service, pending["reminder_id"])
            with service.engine.begin() as connection:
                from app.reminders import _REMINDERS
                connection.execute(_REMINDERS.update().where(_REMINDERS.c.id == pending["reminder_id"])
                    .values(expires_utc=self.due_after(-1).isoformat()))
            service.set_muted("owner", True)
            self.assertIsNone(service.claim_next("owner"))
            rows = service.list_reminders("owner", include_inactive=True)
            self.assertTrue(all(row["status"] == "expired" for row in rows))

    def test_additive_migration_preserves_legacy_claim_and_text(self):
        import sqlite3
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE assistant_reminders (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL, due_utc TEXT NOT NULL, timezone TEXT NOT NULL, status TEXT NOT NULL, created_utc TEXT NOT NULL)")
            db.execute("CREATE TABLE assistant_notifications (id TEXT PRIMARY KEY, reminder_id TEXT NOT NULL UNIQUE, user_id TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL, created_utc TEXT NOT NULL, claimed_utc TEXT, played_utc TEXT, delivery_attempts INTEGER NOT NULL)")
            db.execute("INSERT INTO assistant_reminders VALUES (?,?,?,?,?,?,?)", ("legacy", "owner", "保留的旧提醒", now, "Asia/Shanghai", "fired", now))
            db.execute("INSERT INTO assistant_notifications VALUES (?,?,?,?,?,?,?,?,?)", ("reminder:legacy:due", "legacy", "owner", "保留的旧提醒", "claimed", now, now, None, 1))
        with ReminderService(self.database) as service:
            notice = service.claim_next("owner")
            self.assertEqual(notice["text"], "保留的旧提醒")
            self.assertEqual(notice["delivery_attempts"], 2)
            self.assertEqual(notice["failure_count"], 0)
            self.assertTrue(notice["attempt_id"])
            self.assertTrue(service.begin_output(notice["id"], attempt_id=notice["attempt_id"], channel="text"))
            self.assertTrue(service.acknowledge_delivery(notice["id"], attempt_id=notice["attempt_id"], evidence="fixture-text"))


if __name__ == "__main__":
    unittest.main()
