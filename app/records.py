"""SQLAlchemy records for delivered conversation text and explicit local notes.

Mem0 remains the fact/vector store. These tables store application evidence,
retention, and note transactions; they do not infer facts or run conversations.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import uuid

from sqlalchemy import (Column, Integer, MetaData, String, Table, Text,
                        UniqueConstraint, create_engine, delete, event, insert, select, update)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

META = MetaData()
TURNS = Table("turns", META, Column("id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True), Column("session_id", String, nullable=False),
    Column("created_at", String, nullable=False), Column("status", String, nullable=False),
    Column("user_text", Text, nullable=False), Column("assistant_text", Text, nullable=False, default=""),
    Column("output_channel", String, nullable=False, default="text"))
REFS = Table("turn_memory_refs", META, Column("turn_id", String, primary_key=True),
             Column("memory_id", String, primary_key=True))
DELETED = Table("deleted_memory_refs", META, Column("user_id", String, primary_key=True),
                Column("memory_id", String, primary_key=True))
NOTES = Table("notes", META, Column("id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True), Column("title", String, nullable=False),
    Column("body", Text, nullable=False), Column("source_id", String, nullable=False),
    Column("created_at", String, nullable=False), Column("request_id", String, nullable=False),
    Column("fingerprint", String, nullable=False), UniqueConstraint("user_id", "request_id"))


def now():
    return datetime.now(timezone.utc)


class Records:
    def __init__(self, path, *, retention_days=7):
        if not 1 <= retention_days <= 365:
            raise ValueError("Transcript retention must be 1..365 days")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.gate = threading.RLock()
        self.owner = open(str(self.path) + ".owner.lock", "a+b")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError("Another process owns the conversation records") from None
        self.engine = create_engine("sqlite:///" + str(self.path), connect_args={"timeout": 10, "check_same_thread": False})
        @event.listens_for(self.engine, "connect")
        def configure(connection, _record):
            for pragma in ("journal_mode=WAL", "synchronous=FULL", "secure_delete=ON", "busy_timeout=10000"):
                connection.execute("PRAGMA " + pragma)
        META.create_all(self.engine)
        with self.engine.begin() as db:
            db.execute(update(TURNS).where(TURNS.c.status == "pending").values(status="interrupted"))
        self.expire()

    def close(self):
        self.engine.dispose()
        self.owner.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def expire(self):
        cutoff = (now() - timedelta(days=self.retention_days)).isoformat()
        with self.gate, self.engine.begin() as db:
            stale = select(TURNS.c.id).where(TURNS.c.created_at < cutoff)
            db.execute(delete(REFS).where(REFS.c.turn_id.in_(stale)))
            db.execute(delete(TURNS).where(TURNS.c.created_at < cutoff))

    def begin(self, user_id, session_id, turn_id, text, *, output_channel="text"):
        if not user_id or len(user_id) > 128 or not text.strip() or len(text.encode()) > 8192:
            raise ValueError("Invalid owner or input text exceeds 8 KiB")
        if output_channel not in {"text", "audio", "fixture"}: raise ValueError("Unknown output channel")
        self.expire()
        with self.gate, self.engine.begin() as db:
            db.execute(insert(TURNS).values(id=turn_id, user_id=user_id, session_id=session_id,
                created_at=now().isoformat(), status="pending", user_text=text, assistant_text="", output_channel=output_channel))

    def finish(self, user_id, turn_id, text, *, status="completed"):
        if status not in {"completed", "cancelled", "failed"} or len(text.encode()) > 16384:
            raise ValueError("Invalid output status or text bound")
        with self.gate, self.engine.begin() as db:
            row = db.execute(update(TURNS).where(TURNS.c.id == turn_id, TURNS.c.user_id == user_id,
                TURNS.c.status == "pending").values(status=status, assistant_text=text if status == "completed" else ""))
            return row.rowcount == 1

    def attach_memories(self, user_id, turn_id, memory_ids):
        with self.gate, self.engine.begin() as db:
            owned = db.execute(select(TURNS.c.id).where(TURNS.c.id == turn_id,
                TURNS.c.user_id == user_id, TURNS.c.status == "pending")).first()
            if not owned:
                return False
            for memory_id in memory_ids:
                if db.execute(select(DELETED).where(DELETED.c.user_id == user_id, DELETED.c.memory_id == memory_id)).first():
                    db.execute(update(TURNS).where(TURNS.c.id == turn_id).values(status="redacted", user_text="", assistant_text=""))
                    return False
                db.execute(sqlite_insert(REFS).values(turn_id=turn_id, memory_id=memory_id).on_conflict_do_nothing())
            return True

    def purge_memory(self, user_id, memory_id, *, source_ids=(), deleted=True, exclude_turn=None):
        """Durably redact affected turns before acknowledging memory maintenance."""
        with self.gate, self.engine.begin() as db:
            linked = select(REFS.c.turn_id).where(REFS.c.memory_id == memory_id)
            clause = TURNS.c.id.in_(linked) | TURNS.c.id.in_(list(source_ids))
            statement = update(TURNS).where(TURNS.c.user_id == user_id, clause)
            if exclude_turn:
                statement = statement.where(TURNS.c.id != exclude_turn)
            count = db.execute(statement.values(status="redacted", user_text="", assistant_text="")).rowcount
            if deleted:
                db.execute(sqlite_insert(DELETED).values(user_id=user_id, memory_id=memory_id).on_conflict_do_nothing())
        with self.engine.connect() as db:
            db.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        return count

    def recent(self, user_id, *, session_id=None, limit=12):
        if not 1 <= limit <= 100:
            raise ValueError("History limit must be 1..100")
        cutoff = (now() - timedelta(days=self.retention_days)).isoformat()
        statement = select(TURNS).where(TURNS.c.user_id == user_id, TURNS.c.status == "completed", TURNS.c.created_at >= cutoff)
        if session_id is not None:
            statement = statement.where(TURNS.c.session_id == session_id)
        with self.gate, self.engine.connect() as db:
            rows = list(db.execute(statement.order_by(TURNS.c.created_at.desc()).limit(limit)).mappings())
            return [dict(row) for row in reversed(rows)]

    def memory_refs(self, user_id, turn_ids):
        with self.gate, self.engine.connect() as db:
            return list(db.execute(select(REFS.c.memory_id).join(TURNS, TURNS.c.id == REFS.c.turn_id)
                .where(TURNS.c.user_id == user_id, TURNS.c.id.in_(turn_ids)).distinct()).scalars())

    def filter_memories(self, user_id, items):
        # The transcript barrier is committed before calling Mem0 forget. If
        # that later operation fails, its old payload still cannot re-enter the
        # conversation through a fresh SDK search on a subsequent process.
        with self.gate, self.engine.connect() as db:
            blocked = set(db.execute(select(DELETED.c.memory_id).where(DELETED.c.user_id == user_id,
                DELETED.c.memory_id.in_([item["memory_id"] for item in items]))).scalars())
        return [item for item in items if item["memory_id"] not in blocked]

    def create_note(self, user_id, title, body, *, source_id, request_id):
        if not title.strip() or len(title) > 120 or not body.strip() or len(body.encode()) > 4096:
            raise ValueError("Note title/body exceeds its bound or is empty")
        fingerprint = hashlib.sha256(json.dumps([title, body], ensure_ascii=False).encode()).hexdigest()
        with self.gate, self.engine.begin() as db:
            existing = db.execute(select(NOTES).where(NOTES.c.user_id == user_id, NOTES.c.request_id == request_id)).mappings().first()
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise ValueError("CONFLICT: note request ID has different content")
                note_id = existing["id"]
            else:
                note_id = uuid.uuid4().hex
                db.execute(insert(NOTES).values(id=note_id, user_id=user_id, title=title, body=body,
                    source_id=source_id, created_at=now().isoformat(), request_id=request_id, fingerprint=fingerprint))
            return {"note_id": note_id, "durable": True, "user_message": "笔记“" + title + "”已保存。"}

    def list_notes(self, user_id, *, limit=5):
        if not 1 <= limit <= 100:
            raise ValueError("Note limit must be 1..100")
        with self.gate, self.engine.connect() as db:
            return [dict(row) for row in db.execute(select(NOTES.c.id, NOTES.c.title, NOTES.c.body, NOTES.c.created_at)
                .where(NOTES.c.user_id == user_id).order_by(NOTES.c.created_at.desc()).limit(limit)).mappings()]
