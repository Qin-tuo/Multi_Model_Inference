"""Durable notes and delivered-transcript retention using real SQLite."""
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from sqlalchemy import select, update
from app.records import Records, TURNS, now


class RecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "records.sqlite"
        self.records = Records(self.path)

    def tearDown(self):
        self.records.close()
        self.temp.cleanup()

    def turn(self, turn_id, user="owner", memory_id=None):
        self.records.begin(user, "session", turn_id, "synthetic input " + turn_id)
        if memory_id: self.records.attach_memories(user, turn_id, [memory_id])
        self.records.finish(user, turn_id, "actually delivered " + turn_id)

    def test_note_transaction_replay_and_owner_isolation_survive_reopen(self):
        kwargs = dict(source_id="turn", request_id="request")
        first = self.records.create_note("owner", "采购", "茶叶", **kwargs)
        self.assertEqual(first, self.records.create_note("owner", "采购", "茶叶", **kwargs))
        with self.assertRaises(ValueError): self.records.create_note("owner", "采购", "杯子", **kwargs)
        self.records.close(); self.records = Records(self.path)
        self.assertEqual(len(self.records.list_notes("owner")), 1)
        self.assertEqual(self.records.list_notes("guest"), [])

    def test_only_delivered_completed_turns_enter_history(self):
        self.turn("complete")
        self.records.begin("owner", "session", "cancel", "unanswered")
        self.records.finish("owner", "cancel", "never delivered", status="cancelled")
        self.records.begin("owner", "session", "crashed", "incomplete")
        self.records.close(); self.records = Records(self.path)
        self.assertEqual([r["id"] for r in self.records.recent("owner")], ["complete"])
        self.assertEqual(self.records.recent("guest"), [])

    def test_forget_redacts_sources_and_recalls_but_preserves_unrelated_turns(self):
        self.turn("source", memory_id="memory-1")
        self.turn("recall", memory_id="memory-1")
        self.turn("unrelated", memory_id="memory-2")
        self.turn("other-user", user="guest", memory_id="memory-1")
        self.records.begin("owner", "session", "late", "late source")
        self.records.attach_memories("owner", "late", ["memory-1"])
        self.records.purge_memory("owner", "memory-1")
        self.assertFalse(self.records.finish("owner", "late", "late forgotten content"))
        self.assertEqual([r["id"] for r in self.records.recent("owner")], ["unrelated"])
        self.assertEqual(len(self.records.recent("guest")), 1)
        self.records.begin("owner", "session", "restored", "restored old evidence")
        self.assertFalse(self.records.attach_memories("owner", "restored", ["memory-1"]))
        self.records.close(); self.records = Records(self.path)
        self.assertEqual(self.records.filter_memories("owner", [{"memory_id": "memory-1", "content": "restored SDK payload"}]), [])
        self.assertEqual(len(self.records.filter_memories("guest", [{"memory_id": "memory-1"}])), 1)
        with self.records.engine.connect() as db:
            rows = db.execute(select(TURNS).where(TURNS.c.user_id == "owner", TURNS.c.status == "redacted")).mappings()
            self.assertTrue(all(not row["user_text"] and not row["assistant_text"] for row in rows))

    def test_correction_redacts_previous_use_but_allows_new_version(self):
        self.turn("old", memory_id="memory-1")
        self.records.begin("owner", "session", "new", "new fact")
        self.records.purge_memory("owner", "memory-1", deleted=False, exclude_turn="new")
        self.assertTrue(self.records.attach_memories("owner", "new", ["memory-1"]))
        self.assertTrue(self.records.finish("owner", "new", "new delivered fact"))
        self.assertEqual([r["id"] for r in self.records.recent("owner")], ["new"])

    def test_expiry_and_single_writer(self):
        self.turn("old")
        with self.records.engine.begin() as db:
            db.execute(update(TURNS).where(TURNS.c.id == "old").values(created_at=(now() - timedelta(days=8)).isoformat()))
        self.records.expire()
        self.assertEqual(self.records.recent("owner"), [])
        with self.assertRaises(RuntimeError): Records(self.path)


if __name__ == "__main__": unittest.main()
