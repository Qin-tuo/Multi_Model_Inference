"""Persistent reminders using APScheduler and SQLAlchemy, with a playback outbox.

APScheduler owns timing and restart persistence. This adapter records durable
business IDs, output attempts, expiration and confirmations from output adapters.
DateTrigger uses UTC wall time. This module does not claim acoustic playback or
monotonic-clock protection against system clock changes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.jobstores.base import ConflictingIdError, JobLookupError
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from sqlalchemy import (
    Column, Integer, MetaData, String, Table, Text, and_, create_engine, event,
    func, insert, inspect, select, update,
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
    Column("expires_utc", String(40)),
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
    Column("failure_count", Integer, nullable=False, default=0),
    Column("attempt_id", String(64)),
    Column("output_started_utc", String(40)),
    Column("output_channel", String(24)),
    Column("delivery_evidence", String(256)),
    Column("next_attempt_utc", String(40)),
    Column("delivered_utc", String(40)),
)
_OPERATIONS = Table(
    "assistant_reminder_operations", _METADATA,
    Column("id", String(640), primary_key=True),
    Column("fingerprint", String(64), nullable=False),
    Column("result_json", Text, nullable=False),
)
_POLICY = Table("assistant_reminder_policy", _METADATA,
    Column("user_id", String(256), primary_key=True), Column("muted", Integer, nullable=False))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _engine(database_path: str):
    engine = create_engine(
        "sqlite:///" + database_path,
        connect_args={"timeout": 10, "check_same_thread": False},
    )
    @event.listens_for(engine, "connect")
    def configure(connection, _record):
        for pragma in ("journal_mode=WAL", "synchronous=FULL", "busy_timeout=10000"):
            connection.execute("PRAGMA " + pragma)
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
            if reminder["expires_utc"] and reminder["expires_utc"] <= _utc_now():
                connection.execute(update(_REMINDERS).where(_REMINDERS.c.id == reminder_id,
                    _REMINDERS.c.status == "active").values(status="expired"))
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

    def __init__(self, database_path: Path | str, *, max_pending: int = 100,
                 notification_ttl_seconds: int = 3600, max_delivery_attempts: int = 3,
                 retry_delay_seconds: float = 5):
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.database_path = str(Path(database_path).resolve())
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.max_pending = max_pending
        if not 1 <= notification_ttl_seconds <= 86400 or not 1 <= max_delivery_attempts <= 10 or retry_delay_seconds < 0:
            raise ValueError("Invalid notification TTL or retry limits")
        self.notification_ttl_seconds = notification_ttl_seconds
        self.max_delivery_attempts = max_delivery_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._lock = threading.RLock()
        self._closed = False
        self._live_attempts = {}
        self._started_attempts = set()
        self.visual_guard = None
        self._muted_profiles = {}
        self._owner_file = open(self.database_path + ".owner.lock", "a+b")
        try:
            fcntl.flock(self._owner_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner_file.close()
            raise RuntimeError("another ReminderService owns this database") from None
        try:
            self.engine = _engine(self.database_path)
            _METADATA.create_all(self.engine)
            self._migrate_delivery_columns()
            with self.engine.connect() as connection:
                self._muted_profiles = {row.user_id: bool(row.muted) for row in connection.execute(select(_POLICY))}
            self.scheduler = BackgroundScheduler(
                timezone=timezone.utc,
                jobstores={"default": SQLAlchemyJobStore(engine=self.engine)},
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

    def _migrate_delivery_columns(self):
        # Additive, fixed-schema migration: existing reminder text and jobs stay
        # intact. No user-controlled names or SQL fragments are accepted.
        additions = {_REMINDERS.name: {"expires_utc": "VARCHAR(40)"}, _NOTIFICATIONS.name: {
            "attempt_id": "VARCHAR(64)", "output_started_utc": "VARCHAR(40)", "output_channel": "VARCHAR(24)",
            "delivery_evidence": "VARCHAR(256)", "next_attempt_utc": "VARCHAR(40)", "delivered_utc": "VARCHAR(40)",
            "failure_count": "INTEGER NOT NULL DEFAULT 0"}}
        for table, columns in additions.items():
            existing = {column["name"] for column in inspect(self.engine).get_columns(table)}
            with self.engine.begin() as connection:
                for name, sql_type in columns.items():
                    if name not in existing:
                        connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

    def _recover(self) -> None:
        # The process ownership lock proves the previous consumer is gone.
        # An unacknowledged playback claim is retried with the same notice ID.
        with self.engine.begin() as connection:
            # A crash after output began but before durable acknowledgment has
            # an unknown delivery outcome. Never automatically play it again.
            uncertain = select(_NOTIFICATIONS.c.reminder_id).where(_NOTIFICATIONS.c.status == "playing")
            connection.execute(update(_REMINDERS).where(_REMINDERS.c.id.in_(uncertain)).values(status="uncertain"))
            connection.execute(update(_NOTIFICATIONS).where(_NOTIFICATIONS.c.status == "playing")
                               .values(status="uncertain", delivery_evidence="process_restart_before_output_confirmation"))
            connection.execute(
                update(_NOTIFICATIONS).where(_NOTIFICATIONS.c.status == "claimed")
                .values(status="pending", claimed_utc=None, attempt_id=None)
            )
            # Gesture events belong to the old live scene. They never replay
            # after a consumer restart; partially written output stays uncertain.
            connection.execute(update(_NOTIFICATIONS).where(_NOTIFICATIONS.c.id.startswith('visual:'),
                _NOTIFICATIONS.c.status == 'pending').values(status='expired', delivery_evidence='visual_process_restart'))
            connection.execute(update(_REMINDERS).where(_REMINDERS.c.id.startswith('visual:'),
                _REMINDERS.c.status == 'fired').values(status='expired'))
            legacy = connection.execute(select(_REMINDERS).where(_REMINDERS.c.expires_utc.is_(None))).mappings().all()
            for row in legacy:
                expiry = datetime.fromisoformat(row["due_utc"]) + timedelta(seconds=self.notification_ttl_seconds)
                connection.execute(update(_REMINDERS).where(_REMINDERS.c.id == row["id"]).values(expires_utc=expiry.isoformat()))
            reminders = connection.execute(select(_REMINDERS)).mappings().all()
        active = {item["id"]: item for item in reminders if item["status"] == "active"}
        for job in self.scheduler.get_jobs():
            if job.id.startswith("reminder:") and job.id.removeprefix("reminder:") not in active:
                self.scheduler.remove_job(job.id)
        for reminder in active.values():
            self._ensure_job(reminder)

    def _ensure_job(self, reminder: Any) -> None:
        job_id = "reminder:" + reminder["id"]
        existing = self.scheduler.get_job(job_id)
        due_at = datetime.fromisoformat(reminder['due_utc'])
        # Persisted APScheduler args include the database's absolute path.
        # Rebind while recovery is paused if a backup was restored elsewhere;
        # otherwise its callback would mutate the old database (if it exists).
        valid = (existing is not None and existing.func_ref == 'app.reminders:_deliver_due'
                 and tuple(existing.args) == (self.database_path, reminder['id']) and not existing.kwargs
                 and isinstance(existing.trigger, DateTrigger) and existing.trigger.run_date == due_at)
        if valid:
            return
        try:
            self.scheduler.add_job(
                _deliver_due,
                trigger=DateTrigger(run_date=due_at),
                args=(self.database_path, reminder["id"]),
                id=job_id,
                name="local-reminder",
                replace_existing=existing is not None,
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
                            and_(_REMINDERS.c.user_id == user_id, ~_REMINDERS.c.id.startswith('visual:'),
                                 _REMINDERS.c.status.in_(["active", "fired", "uncertain"]))
                        )
                    ).scalar_one()
                    if outstanding >= self.max_pending:
                        raise RuntimeError("OVERLOADED: reminder pending limit reached")
                    reminder = {
                        "id": uuid.uuid4().hex, "user_id": user_id, "text": text,
                        "due_utc": due_utc, "timezone": timezone_name,
                        "status": "active", "created_utc": _utc_now(),
                        "expires_utc": (due_at.astimezone(timezone.utc) + timedelta(seconds=self.notification_ttl_seconds)).isoformat(),
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
                    notice_id = reminder_id if reminder_id.startswith('visual:') else f'reminder:{reminder_id}:due'
                    self._clear_attempt(notice_id, self._live_attempts.get(notice_id))
                    connection.execute(update(_REMINDERS).where(_REMINDERS.c.id == reminder_id).values(status="cancelled"))
                    connection.execute(update(_NOTIFICATIONS).where(and_(
                        _NOTIFICATIONS.c.reminder_id == reminder_id,
                        _NOTIFICATIONS.c.status.in_(["pending", "claimed", "playing", "uncertain", "failed"]),
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
            statement = select(_REMINDERS).where(_REMINDERS.c.user_id == user_id,
                                                ~_REMINDERS.c.id.startswith('visual:'))
            if not include_inactive:
                statement = statement.where(_REMINDERS.c.status.in_(["active", "fired", "uncertain", "failed"]))
            with self.engine.connect() as connection:
                return [dict(row) for row in connection.execute(statement.order_by(_REMINDERS.c.due_utc)).mappings()]

    def claim_next(self, user_id: str, *, muted: bool = False, occupied: bool = False) -> dict[str, Any] | None:
        """Claim one notice only when Pipecat can play; no audio is emitted here."""
        _validate_id(user_id, "user_id")
        with self._lock:
            self._check_open()
            with self.engine.begin() as connection:
                expired = list(connection.execute(select(_REMINDERS.c.id).where(_REMINDERS.c.user_id == user_id,
                    _REMINDERS.c.status == "fired", _REMINDERS.c.expires_utc <= _utc_now(),
                    _REMINDERS.c.id.in_(select(_NOTIFICATIONS.c.reminder_id).where(_NOTIFICATIONS.c.status == "pending")))).scalars())
                connection.execute(update(_NOTIFICATIONS).where(_NOTIFICATIONS.c.reminder_id.in_(expired),
                    _NOTIFICATIONS.c.status == "pending").values(status="expired"))
                connection.execute(update(_REMINDERS).where(_REMINDERS.c.id.in_(expired)).values(status="expired"))
            if muted or occupied or self.is_muted(user_id):
                return None
            candidate = select(_NOTIFICATIONS.c.id).join(
                _REMINDERS, _REMINDERS.c.id == _NOTIFICATIONS.c.reminder_id
            ).where(and_(
                _NOTIFICATIONS.c.user_id == user_id,
                _NOTIFICATIONS.c.status == "pending",
                _REMINDERS.c.status == "fired",
                _REMINDERS.c.expires_utc > _utc_now(),
                (_NOTIFICATIONS.c.next_attempt_utc.is_(None) | (_NOTIFICATIONS.c.next_attempt_utc <= _utc_now())),
                _NOTIFICATIONS.c.failure_count < self.max_delivery_attempts,
            )).order_by(_NOTIFICATIONS.c.created_utc, _NOTIFICATIONS.c.id).limit(1).scalar_subquery()
            with self.engine.begin() as connection:
                row = connection.execute(update(_NOTIFICATIONS).where(and_(
                    _NOTIFICATIONS.c.id == candidate, _NOTIFICATIONS.c.status == "pending",
                )).values(
                    status="claimed", claimed_utc=_utc_now(),
                    delivery_attempts=_NOTIFICATIONS.c.delivery_attempts + 1,
                    attempt_id=uuid.uuid4().hex, output_started_utc=None, output_channel=None, delivery_evidence=None,
                ).returning(*_NOTIFICATIONS.c)).mappings().first()
                if row is None: return None
                notice = dict(row)
                notice["expires_utc"] = connection.execute(select(_REMINDERS.c.expires_utc)
                    .where(_REMINDERS.c.id == notice["reminder_id"])).scalar_one()
            self._live_attempts[notice["id"]] = notice["attempt_id"]
            return notice

    def is_muted(self, user_id: str) -> bool:
        _validate_id(user_id, "user_id")
        self._check_open()
        return self._muted_profiles.get(user_id, False)

    def set_muted(self, user_id: str, muted: bool) -> None:
        _validate_id(user_id, "user_id")
        if not isinstance(muted, bool): raise ValueError("muted must be boolean")
        with self._lock:
            self._check_open()
            with self.engine.begin() as connection:
                connection.execute(sqlite_insert(_POLICY).values(user_id=user_id, muted=int(muted))
                    .on_conflict_do_update(index_elements=["user_id"], set_={"muted": int(muted)}))
            self._muted_profiles[user_id] = muted

    def attempt_active(self, notice) -> bool:
        """Fast audio-path revocation check; this service is the sole DB owner.

        Durable checks still happen at begin/ack. The output loop never waits
        on a database lock merely to decide whether an old PCM chunk is valid.
        """
        return (not self._closed and self._live_attempts.get(notice["id"]) == notice["attempt_id"]
                and not self._muted_profiles.get(notice["user_id"], False)
                and self._visual_allowed(notice['id'], started=notice['attempt_id'] in self._started_attempts))

    def _visual_allowed(self, notification_id, *, started=False):
        return (not notification_id.startswith('visual:') or
                bool(self.visual_guard and self.visual_guard(notification_id, started=started)))

    def _clear_attempt(self, notification_id, attempt_id):
        self._started_attempts.discard(attempt_id)
        if self._live_attempts.get(notification_id) == attempt_id:
            self._live_attempts.pop(notification_id, None)

    def notification_allowed(self, notification_id: str, attempt_id: str | None = None) -> bool:
        """Recheck immediately before playback; a later cancel blocks old claims."""
        with self._lock:
            self._check_open()
            token = attempt_id or self._live_attempts.get(notification_id)
            if not self._visual_allowed(notification_id, started=token in self._started_attempts): return False
            with self.engine.connect() as connection:
                statement = select(_NOTIFICATIONS.c.id).join(
                    _REMINDERS, _REMINDERS.c.id == _NOTIFICATIONS.c.reminder_id
                ).where(and_(
                    _NOTIFICATIONS.c.id == notification_id,
                    _NOTIFICATIONS.c.status.in_(["claimed", "playing"]), _REMINDERS.c.status == "fired",
                    ((_NOTIFICATIONS.c.status == "playing") | (_REMINDERS.c.expires_utc > _utc_now())),
                ))
                if attempt_id is not None: statement = statement.where(_NOTIFICATIONS.c.attempt_id == attempt_id)
                return connection.execute(statement).first() is not None

    def release_notification(self, notification_id: str, *, attempt_id: str) -> bool:
        """Return an unplayed claim to the outbox with the same notification ID."""
        with self._lock:
            self._check_open()
            with self.engine.begin() as connection:
                result = connection.execute(update(_NOTIFICATIONS).where(and_(
                    _NOTIFICATIONS.c.id == notification_id, _NOTIFICATIONS.c.status == "claimed",
                    _NOTIFICATIONS.c.attempt_id == attempt_id,
                )).values(status="pending", claimed_utc=None, attempt_id=None))
                if result.rowcount == 1: self._clear_attempt(notification_id, attempt_id)
                return result.rowcount == 1

    def release_unwritten_output(self, notification_id: str, *, attempt_id: str) -> bool:
        """Only the live output adapter can prove no device write was attempted."""
        with self._lock:
            self._check_open()
            with self.engine.begin() as connection:
                changed = connection.execute(update(_NOTIFICATIONS).where(_NOTIFICATIONS.c.id == notification_id,
                    _NOTIFICATIONS.c.attempt_id == attempt_id, _NOTIFICATIONS.c.status == "playing")
                    .values(status="pending", attempt_id=None, output_started_utc=None, output_channel=None,
                            claimed_utc=None, delivery_evidence="deferred_before_any_output_write"))
            if changed.rowcount: self._clear_attempt(notification_id, attempt_id)
            return changed.rowcount == 1

    def begin_output(self, notification_id: str, *, attempt_id: str, channel: str) -> bool:
        if channel not in {"audio", "text", "fixture"}: raise ValueError("Unknown output channel")
        with self._lock:
            if not self.notification_allowed(notification_id, attempt_id): return False
            with self.engine.begin() as connection:
                changed = connection.execute(update(_NOTIFICATIONS).where(_NOTIFICATIONS.c.id == notification_id,
                    _NOTIFICATIONS.c.attempt_id == attempt_id, _NOTIFICATIONS.c.status == "claimed")
                    .values(status="playing", output_started_utc=_utc_now(), output_channel=channel))
                if changed.rowcount == 1: self._started_attempts.add(attempt_id)
                return changed.rowcount == 1

    def fail_output(self, notification_id: str, *, attempt_id: str, reason: str) -> bool:
        """Retry only when no output was admitted; partial/unknown output stays visible."""
        with self._lock, self.engine.begin() as connection:
            self._check_open()
            row = connection.execute(select(_NOTIFICATIONS).where(_NOTIFICATIONS.c.id == notification_id,
                _NOTIFICATIONS.c.attempt_id == attempt_id)).mappings().first()
            if row is None or row["status"] not in {"claimed", "playing"}: return False
            failures = row["failure_count"] + 1
            status = "uncertain" if row["status"] == "playing" else ("failed" if failures >= self.max_delivery_attempts else "pending")
            connection.execute(update(_NOTIFICATIONS).where(_NOTIFICATIONS.c.id == notification_id)
                .values(status=status, delivery_evidence=reason[:256], claimed_utc=None, failure_count=failures,
                        next_attempt_utc=(datetime.now(timezone.utc) + timedelta(seconds=self.retry_delay_seconds)).isoformat()))
            if status != "pending":
                connection.execute(update(_REMINDERS).where(_REMINDERS.c.id == row["reminder_id"]).values(status=status))
            self._clear_attempt(notification_id, attempt_id)
            return True

    def acknowledge_delivery(self, notification_id: str, *, attempt_id: str, evidence: str) -> bool:
        """Confirm the same started attempt after its output sink has drained."""
        if not evidence or len(evidence) > 256: raise ValueError("Delivery evidence is required")
        with self._lock:
            self._check_open()
            with self.engine.begin() as connection:
                notification = connection.execute(select(_NOTIFICATIONS).where(
                    _NOTIFICATIONS.c.id == notification_id
                )).mappings().first()
                if notification is None or notification["attempt_id"] != attempt_id:
                    return False
                if notification["status"] in {"played", "displayed", "fixture_delivered"}:
                    return True
                if notification["status"] != "playing":
                    return False
                if not self._visual_allowed(notification_id, started=True): return False
                delivered_status = {"audio": "played", "text": "displayed", "fixture": "fixture_delivered"}[notification["output_channel"]]
                changed = connection.execute(update(_NOTIFICATIONS).where(and_(
                    _NOTIFICATIONS.c.id == notification_id, _NOTIFICATIONS.c.status == "playing",
                    _NOTIFICATIONS.c.attempt_id == attempt_id,
                )).values(status=delivered_status, delivered_utc=_utc_now(), delivery_evidence=evidence,
                          played_utc=_utc_now() if notification["output_channel"] == "audio" else None))
                if changed.rowcount != 1:
                    return False
                connection.execute(update(_REMINDERS).where(and_(
                    _REMINDERS.c.id == notification["reminder_id"], _REMINDERS.c.status == "fired",
                )).values(status="delivered"))
                self._clear_attempt(notification_id, attempt_id)
                return True
