"""Persistent reminders using APScheduler and SQLAlchemy, with a playback outbox.

APScheduler owns timing and restart persistence. This adapter records durable
business IDs and notification delivery state; audio is consumed by Pipecat later.
DateTrigger uses UTC wall time. This module does not claim acoustic playback or
monotonic-clock protection against system clock changes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.jobstores.base import ConflictingIdError, JobLookupError
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from sqlalchemy import (
    Column, Integer, MetaData, String, Table, Text, and_, create_engine,
    func, insert, select, update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

_METADATA = MetaData()
_REMINDERS = Table(
    "assistant_reminders", _METADATA,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(256), nullable=False, index=True),
    Column("text", Text, nullable=False),
    Column("due_utc", String(40), nullable=False),
    Column("timezone", String(128), nullable=False),
    Column("status", String(24), nullable=False),
    Column("created_utc", String(40), nullable=False),
)
_NOTIFICATIONS = Table(
    "assistant_notifications", _METADATA,
    Column("id", String(96), primary_key=True),
    Column("reminder_id", String(64), nullable=False, unique=True),
    Column("user_id", String(256), nullable=False, index=True),
    Column("text", Text, nullable=False),
    Column("status", String(24), nullable=False),
    Column("created_utc", String(40), nullable=False),
    Column("claimed_utc", String(40)),
    Column("played_utc", String(40)),
    Column("delivery_attempts", Integer, nullable=False, default=0),
)
_OPERATIONS = Table(
    "assistant_reminder_operations", _METADATA,
    Column("id", String(640), primary_key=True),
    Column("fingerprint", String(64), nullable=False),
    Column("result_json", Text, nullable=False),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _engine(database_path: str):
    engine = create_engine(
        "sqlite:///" + database_path,
        connect_args={"timeout": 10, "check_same_thread": False},
    )
    return engine


def _deliver_due(database_path: str, reminder_id: str) -> None:
    """Pickle-safe APScheduler callback; enqueue one durable logical notice."""
    engine = _engine(database_path)
    try:
        with engine.begin() as connection:
            reminder = connection.execute(
                select(_REMINDERS).where(_REMINDERS.c.id == reminder_id)
            ).mappings().first()
            if reminder is None or reminder["status"] != "active":
                return
            changed = connection.execute(
                update(_REMINDERS)
                .where(and_(_REMINDERS.c.id == reminder_id, _REMINDERS.c.status == "active"))
                .values(status="fired")
            )
            if changed.rowcount != 1:
                return
            connection.execute(
                sqlite_insert(_NOTIFICATIONS).values(
                    id=f"reminder:{reminder_id}:due", reminder_id=reminder_id,
                    user_id=reminder["user_id"], text=reminder["text"],
                    status="pending", created_utc=_utc_now(), delivery_attempts=0,
                ).on_conflict_do_nothing(index_elements=["reminder_id"])
            )
    finally:
        engine.dispose()


def _validate_id(value: str, label: str) -> None:
    if not value or len(value) > 256:
        raise ValueError(f"{label} must contain 1–256 characters")


class ReminderService:
    """Single-owner APScheduler plugin with bounded pending work and CRUD tools."""

    def __init__(self, database_path: Path | str, *, max_pending: int = 100):
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.database_path = str(Path(database_path).resolve())
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.max_pending = max_pending
        self._lock = threading.RLock()
        self._closed = False
        self._owner_file = open(self.database_path + ".owner.lock", "a+b")
        try:
            fcntl.flock(self._owner_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner_file.close()
            raise RuntimeError("another ReminderService owns this database") from None
        try:
            self.engine = _engine(self.database_path)
            _METADATA.create_all(self.engine)
            self.scheduler = BackgroundScheduler(
                timezone=timezone.utc,
                jobstores={"default": SQLAlchemyJobStore(url="sqlite:///" + self.database_path)},
                job_defaults={"misfire_grace_time": None, "coalesce": True, "max_instances": 1},
                daemon=True,
            )
            self.scheduler.start(paused=True)
            self._recover()
            self.scheduler.resume()
        except Exception:
            if hasattr(self, "scheduler") and self.scheduler.running:
                self.scheduler.shutdown(wait=True)
            if hasattr(self, "engine"):
                self.engine.dispose()
            self._owner_file.close()
            raise

    def __enter__(self) -> "ReminderService":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.scheduler.shutdown(wait=True)
            self.engine.dispose()
            fcntl.flock(self._owner_file.fileno(), fcntl.LOCK_UN)
            self._owner_file.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("reminder service is closed")

    def _recover(self) -> None:
        # The process ownership lock proves the previous consumer is gone.
        # An unacknowledged playback claim is retried with the same notice ID.
        with self.engine.begin() as connection:
            connection.execute(
                update(_NOTIFICATIONS).where(_NOTIFICATIONS.c.status == "claimed")
                .values(status="pending", claimed_utc=None)
            )
            reminders = connection.execute(select(_REMINDERS)).mappings().all()
        active = {item["id"]: item for item in reminders if item["status"] == "active"}
        for job in self.scheduler.get_jobs():
            if job.id.startswith("reminder:") and job.id.removeprefix("reminder:") not in active:
                self.scheduler.remove_job(job.id)
        for reminder in active.values():
            self._ensure_job(reminder)

    def _ensure_job(self, reminder: Any) -> None:
        job_id = "reminder:" + reminder["id"]
        if self.scheduler.get_job(job_id) is not None:
            return
        try:
            self.scheduler.add_job(
                _deliver_due,
                trigger=DateTrigger(run_date=datetime.fromisoformat(reminder["due_utc"])),
                args=(self.database_path, reminder["id"]),
                id=job_id,
                name="local-reminder",
            )
        except ConflictingIdError:
            pass

    @staticmethod
    def _operation_key(user_id: str, request_id: str, action: str, arguments: dict[str, Any]):
        _validate_id(user_id, "user_id")
        _validate_id(request_id, "request_id")
        payload = json.dumps({"action": action, "arguments": arguments}, ensure_ascii=False, sort_keys=True)
        # Length-prefix user/request IDs so a colon in one cannot alias another.
        key = f"{len(user_id)}:{user_id}{len(request_id)}:{request_id}"
        return key, hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _existing_operation(connection, key: str, fingerprint: str):
        existing = connection.execute(select(_OPERATIONS).where(_OPERATIONS.c.id == key)).mappings().first()
        if existing is None:
            return None
        if existing["fingerprint"] != fingerprint:
            raise ValueError("CONFLICT: request_id was used for different arguments")
        return json.loads(existing["result_json"])

    def create(
        self, user_id: str, text: str, due_at: datetime, *, request_id: str,
        timezone_name: str = "Asia/Shanghai",
    ) -> dict[str, Any]:
        """Commit a reminder first, then let APScheduler own its execution."""
        if not text.strip() or len(text.encode("utf-8")) > 2048:
            raise ValueError("reminder text must fit within 2 KiB")
        if due_at.tzinfo is None:
            raise ValueError("due_at must include an explicit timezone")
        ZoneInfo(timezone_name)
        due_utc = due_at.astimezone(timezone.utc).isoformat()
        key, fingerprint = self._operation_key(user_id, request_id, "create", {
            "text": text, "due_utc": due_utc, "timezone": timezone_name,
        })
        with self._lock:
            self._check_open()
            with self.engine.begin() as connection:
                existing = self._existing_operation(connection, key, fingerprint)
                if existing is not None:
                    result = existing
                    reminder = connection.execute(
                        select(_REMINDERS).where(_REMINDERS.c.id == result["reminder_id"])
                    ).mappings().one()
                else:
                    outstanding = connection.execute(
                        select(func.count()).select_from(_REMINDERS).where(
                            and_(_REMINDERS.c.user_id == user_id, _REMINDERS.c.status.in_(["active", "fired"]))
                        )
                    ).scalar_one()
                    if outstanding >= self.max_pending:
                        raise RuntimeError("OVERLOADED: reminder pending limit reached")
                    reminder = {
                        "id": uuid.uuid4().hex, "user_id": user_id, "text": text,
                        "due_utc": due_utc, "timezone": timezone_name,
                        "status": "active", "created_utc": _utc_now(),
                    }
                    connection.execute(insert(_REMINDERS).values(**reminder))
                    result = {"status": "scheduled", "reminder_id": reminder["id"], "due_utc": due_utc}
                    connection.execute(insert(_OPERATIONS).values(
                        id=key, fingerprint=fingerprint,
                        result_json=json.dumps(result, ensure_ascii=False),
                    ))
            if reminder["status"] == "active":
                self._ensure_job(reminder)
            return result

    def cancel(self, user_id: str, reminder_id: str, *, request_id: str) -> dict[str, Any]:
        """Cancel future dispatch and pending notices; played output stays history."""
        key, fingerprint = self._operation_key(user_id, request_id, "cancel", {"reminder_id": reminder_id})
        with self._lock:
            self._check_open()
            with self.engine.begin() as connection:
                existing = self._existing_operation(connection, key, fingerprint)
                if existing is not None:
                    return existing
                reminder = connection.execute(select(_REMINDERS).where(and_(
                    _REMINDERS.c.id == reminder_id, _REMINDERS.c.user_id == user_id,
                ))).mappings().first()
                if reminder is None:
                    result = {"status": "not_found", "reminder_id": reminder_id}
                else:
                    connection.execute(update(_REMINDERS).where(_REMINDERS.c.id == reminder_id).values(status="cancelled"))
                    connection.execute(update(_NOTIFICATIONS).where(and_(
                        _NOTIFICATIONS.c.reminder_id == reminder_id,
                        _NOTIFICATIONS.c.status.in_(["pending", "claimed"]),
                    )).values(status="cancelled"))
                    result = {"status": "cancelled", "reminder_id": reminder_id, "played_output_rolled_back": False}
                connection.execute(insert(_OPERATIONS).values(
                    id=key, fingerprint=fingerprint, result_json=json.dumps(result),
                ))
            if reminder is not None:
                try:
                    self.scheduler.remove_job("reminder:" + reminder_id)
                except JobLookupError:
                    pass
            return result

    def list_reminders(self, user_id: str, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        _validate_id(user_id, "user_id")
        with self._lock:
            self._check_open()
            statement = select(_REMINDERS).where(_REMINDERS.c.user_id == user_id)
            if not include_inactive:
                statement = statement.where(_REMINDERS.c.status.in_(["active", "fired"]))
            with self.engine.connect() as connection:
                return [dict(row) for row in connection.execute(statement.order_by(_REMINDERS.c.due_utc)).mappings()]

    def claim_next(self, user_id: str, *, muted: bool = False, occupied: bool = False) -> dict[str, Any] | None:
        """Claim one notice only when Pipecat can play; no audio is emitted here."""
        _validate_id(user_id, "user_id")
        if muted or occupied:
            return None
        with self._lock:
            self._check_open()
            candidate = select(_NOTIFICATIONS.c.id).join(
                _REMINDERS, _REMINDERS.c.id == _NOTIFICATIONS.c.reminder_id
            ).where(and_(
                _NOTIFICATIONS.c.user_id == user_id,
                _NOTIFICATIONS.c.status == "pending",
                _REMINDERS.c.status == "fired",
            )).order_by(_NOTIFICATIONS.c.created_utc, _NOTIFICATIONS.c.id).limit(1).scalar_subquery()
            with self.engine.begin() as connection:
                row = connection.execute(update(_NOTIFICATIONS).where(and_(
                    _NOTIFICATIONS.c.id == candidate, _NOTIFICATIONS.c.status == "pending",
                )).values(
                    status="claimed", claimed_utc=_utc_now(),
                    delivery_attempts=_NOTIFICATIONS.c.delivery_attempts + 1,
                ).returning(*_NOTIFICATIONS.c)).mappings().first()
                return dict(row) if row is not None else None

    def notification_allowed(self, notification_id: str) -> bool:
        """Recheck immediately before playback; a later cancel blocks old claims."""
        with self._lock:
            self._check_open()
            with self.engine.connect() as connection:
                return connection.execute(select(_NOTIFICATIONS.c.id).join(
                    _REMINDERS, _REMINDERS.c.id == _NOTIFICATIONS.c.reminder_id
                ).where(and_(
                    _NOTIFICATIONS.c.id == notification_id,
                    _NOTIFICATIONS.c.status == "claimed", _REMINDERS.c.status == "fired",
                ))).first() is not None

    def release_notification(self, notification_id: str) -> bool:
        """Return an unplayed claim to the outbox with the same notification ID."""
        with self._lock:
            self._check_open()
            with self.engine.begin() as connection:
                result = connection.execute(update(_NOTIFICATIONS).where(and_(
                    _NOTIFICATIONS.c.id == notification_id, _NOTIFICATIONS.c.status == "claimed",
                )).values(status="pending", claimed_utc=None))
                return result.rowcount == 1

    def acknowledge_played(self, notification_id: str) -> bool:
        """Call after actual playback confirmation, never merely after TTS submit."""
        with self._lock:
            self._check_open()
            with self.engine.begin() as connection:
                notification = connection.execute(select(_NOTIFICATIONS).where(
                    _NOTIFICATIONS.c.id == notification_id
                )).mappings().first()
                if notification is None:
                    return False
                if notification["status"] == "played":
                    return True
                if notification["status"] != "claimed":
                    return False
                changed = connection.execute(update(_NOTIFICATIONS).where(and_(
                    _NOTIFICATIONS.c.id == notification_id, _NOTIFICATIONS.c.status == "claimed",
                )).values(status="played", played_utc=_utc_now()))
                if changed.rowcount != 1:
                    return False
                connection.execute(update(_REMINDERS).where(and_(
                    _REMINDERS.c.id == notification["reminder_id"], _REMINDERS.c.status == "fired",
                )).values(status="delivered"))
                return True
