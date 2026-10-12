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
            self.assertFalse(service.acknowledge_played(notice["id"]))
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
            self.assertTrue(after_claim_restart.acknowledge_played(notice_id))
            self.assertTrue(after_claim_restart.acknowledge_played(notice_id))
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
            self.assertTrue(service.release_notification(notice["id"]))
            retried = service.claim_next("owner")
            self.assertEqual(retried["id"], notice["id"])
            self.assertEqual(retried["delivery_attempts"], 2)
            # Tests acknowledge the delivery protocol, not an acoustic measurement.
            self.assertTrue(service.acknowledge_played(retried["id"]))
            cancelled = service.cancel("owner", reminder["reminder_id"], request_id="cancel-history")
            self.assertFalse(cancelled["played_output_rolled_back"])
            with service.engine.connect() as connection:
                state = connection.execute(select(_NOTIFICATIONS.c.status)).scalar_one()
                self.assertEqual(state, "played")


if __name__ == "__main__":
    unittest.main()
