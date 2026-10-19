# SPDX-License-Identifier: Apache-2.0
"""Durable finite actions using SQLAlchemy transactions and APScheduler dates.

Adapted from the previous project's SQLAlchemy/ReminderService contracts, with
memory, identity and visual-notice policy removed. Only this optional service
owns its new database; old project databases are never opened or migrated.
"""

from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import re
from pathlib import Path
import threading
import uuid

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
    insert,
    inspect,
    select,
    update,
)
from app.actions.schema import validate


META = MetaData()
REQUESTS = Table(
    "reachy_action_requests",
    META,
    Column("id", String, primary_key=True),
    Column("fingerprint", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("status", String, nullable=False),
)
OPERATIONS = Table(
    "reachy_action_operations",
    META,
    Column("id", String, primary_key=True),
    Column("request_id", String, nullable=False, index=True),
    Column("position", Integer, nullable=False),
    Column("fingerprint", String, nullable=False),
    Column("name", String, nullable=False),
    Column("result", Text, nullable=False),
)
NOTES = Table(
    "reachy_notes",
    META,
    Column("id", String, primary_key=True),
    Column("title", String, nullable=False),
    Column("body", Text, nullable=False),
    Column("created_at", String, nullable=False),
)
REMINDERS = Table(
    "reachy_reminders",
    META,
    Column("id", String, primary_key=True),
    Column("text", Text, nullable=False),
    Column("due_at", String, nullable=False),
    Column("expires_at", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("status", String, nullable=False),
    Column("attempt", String),
    Column("attempts", Integer, nullable=False, default=0),
    Column("next_attempt", String),
    Column("receipt", Text),
)


def now():
    return datetime.now(timezone.utc)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def engine(path):
    result = create_engine(
        "sqlite:///" + str(path),
        connect_args={"timeout": 0.25, "check_same_thread": False},
    )

    @event.listens_for(result, "connect")
    def configure(connection, _):
        for setting in ["journal_mode=WAL", "synchronous=FULL", "busy_timeout=250"]:
            connection.execute("PRAGMA " + setting)

    return result


def mark_due(path, reminder_id):
    """Serializable APScheduler callback: make a notice durable, never play audio."""
    db = engine(path)
    try:
        with db.begin() as connection:
            row = (
                connection.execute(
                    select(REMINDERS).where(REMINDERS.c.id == reminder_id)
                )
                .mappings()
                .first()
            )
            if (
                row
                and row["status"] == "scheduled"
                and row["due_at"] <= now().isoformat()
            ):
                connection.execute(
                    update(REMINDERS)
                    .where(
                        REMINDERS.c.id == reminder_id, REMINDERS.c.status == "scheduled"
                    )
                    .values(
                        status="expired"
                        if row["expires_at"] <= now().isoformat()
                        else "due"
                    )
                )
    finally:
        db.dispose()


class ActionStore:
    def __init__(self, config):
        self.config = config
        self.path = str(Path(config.database).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._closed = False
        self.last_error = None
        self.owner = open(self.path + ".owner.lock", "a+b")
        self.db = None
        self.scheduler = None
        try:
            try:
                fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    "Another application owns this action database"
                ) from None
            self.db = engine(self.path)
            existing = set(inspect(self.db).get_table_names())
            if existing and not existing <= set(META.tables) | {"apscheduler_jobs"}:
                raise ValueError(
                    "Database contains unrelated tables; refusing to modify it"
                )
            META.create_all(self.db)
            with self.db.begin() as c:
                c.execute(
                    update(REQUESTS)
                    .where(REQUESTS.c.status == "running")
                    .values(status="interrupted")
                )
                c.execute(
                    update(REMINDERS)
                    .where(REMINDERS.c.status == "claimed")
                    .values(status="due", attempt=None)
                )
                c.execute(
                    update(REMINDERS)
                    .where(REMINDERS.c.status == "delivering")
                    .values(
                        status="uncertain",
                        receipt=json.dumps(
                            {"kind": "process_restart_before_output_confirmation"}
                        ),
                    )
                )
            self.scheduler = BackgroundScheduler(
                timezone=timezone.utc,
                jobstores={
                    "default": SQLAlchemyJobStore(engine=self.db),
                    "volatile": MemoryJobStore(),
                },
                executors={"default": ThreadPoolExecutor(max_workers=1)},
                job_defaults={
                    "misfire_grace_time": None,
                    "coalesce": True,
                    "max_instances": 1,
                },
                daemon=True,
            )
            self.scheduler.start(paused=True)
            self.reconcile()
            self.scheduler.add_job(
                self.reconcile,
                "interval",
                seconds=1,
                id="action-reconcile",
                jobstore="volatile",
            )
            self.scheduler.resume()
        except Exception:
            self.close()
            raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Never wait for an executor while holding the lock it may be acquiring.
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
        if self.db:
            self.db.dispose()
        self.owner.close()

    def _open(self):
        if self._closed:
            raise RuntimeError("Action store is closed")

    def reconcile(self):
        with self._lock:
            if self._closed:
                return
            try:
                with self.db.begin() as c:
                    c.execute(
                        update(REMINDERS)
                        .where(
                            REMINDERS.c.status.in_(["scheduled", "due", "claimed"]),
                            REMINDERS.c.expires_at <= now().isoformat(),
                        )
                        .values(status="expired")
                    )
                    rows = (
                        c.execute(
                            select(REMINDERS).where(REMINDERS.c.status == "scheduled")
                        )
                        .mappings()
                        .all()
                    )
                active = {row["id"] for row in rows}
                for job in self.scheduler.get_jobs(jobstore="default"):
                    if job.id.startswith("reminder:") and job.id[9:] not in active:
                        self.scheduler.remove_job(job.id)
                for row in rows:
                    key = "reminder:" + row["id"]
                    due = datetime.fromisoformat(row["due_at"])
                    job = self.scheduler.get_job(key)
                    valid = (
                        job
                        and job.func_ref == "app.actions.store:mark_due"
                        and tuple(job.args) == (self.path, row["id"])
                        and not job.kwargs
                        and isinstance(job.trigger, DateTrigger)
                        and job.trigger.run_date == due
                    )
                    if not valid:
                        self.scheduler.add_job(
                            mark_due,
                            DateTrigger(run_date=due),
                            args=(self.path, row["id"]),
                            id=key,
                            replace_existing=True,
                        )
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)

    def _receipts(self, c, key):
        return [
            {
                "name": r["name"],
                "position": r["position"],
                "result": json.loads(r["result"]),
            }
            for r in c.execute(
                select(OPERATIONS)
                .where(OPERATIONS.c.request_id == key)
                .order_by(OPERATIONS.c.position)
            ).mappings()
        ]

    def begin_request(self, key, digest):
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"[0-9a-f]{32}", key)
            or not isinstance(digest, str)
            or not 1 <= len(digest) <= 128
        ):
            raise ValueError("Invalid action request identity")
        with self._lock, self.db.begin() as c:
            self._open()
            row = (
                c.execute(select(REQUESTS).where(REQUESTS.c.id == key))
                .mappings()
                .first()
            )
            if row:
                if row["fingerprint"] != digest:
                    raise ValueError("Request ID was reused for different input")
                return {
                    "replay": True,
                    "status": row["status"],
                    "receipts": self._receipts(c, key),
                }
            if (
                c.scalar(select(func.count()).select_from(REQUESTS))
                >= self.config.max_requests
            ):
                raise RuntimeError("Action request ledger is full")
            c.execute(
                insert(REQUESTS).values(
                    id=key,
                    fingerprint=digest,
                    created_at=now().isoformat(),
                    status="running",
                )
            )
            return {"replay": False}

    def knows_request(self, key, digest):
        with self._lock, self.db.connect() as c:
            self._open()
            row = (
                c.execute(select(REQUESTS).where(REQUESTS.c.id == key))
                .mappings()
                .first()
            )
            if row and row["fingerprint"] != digest:
                raise ValueError("Request ID was reused for different input")
            return row is not None

    def finish_request(self, key, status):
        with self._lock, self.db.begin() as c:
            self._open()
            c.execute(
                update(REQUESTS)
                .where(REQUESTS.c.id == key, REQUESTS.c.status == "running")
                .values(status=status)
            )

    def receipts(self, key):
        with self._lock, self.db.connect() as c:
            return self._receipts(c, key)

    def execute(self, key, position, name, arguments):
        if (
            type(position) is not int
            or not 0 <= position < self.config.max_calls_per_turn
        ):
            raise ValueError("Tool budget exceeded")
        values = validate(name, arguments)
        digest = fingerprint({"name": name, "arguments": values})
        operation = key + ":" + str(position)
        with self._lock, self.db.begin() as c:
            self._open()
            old = (
                c.execute(select(OPERATIONS).where(OPERATIONS.c.id == operation))
                .mappings()
                .first()
            )
            if old:
                if old["fingerprint"] != digest:
                    raise ValueError("Operation key was reused for different arguments")
                return json.loads(old["result"])
            equivalent = (
                c.execute(
                    select(OPERATIONS).where(
                        OPERATIONS.c.request_id == key,
                        OPERATIONS.c.fingerprint == digest,
                    )
                )
                .mappings()
                .first()
            )
            if equivalent:
                return {**json.loads(equivalent["result"]), "replayed": True}
            request = (
                c.execute(select(REQUESTS).where(REQUESTS.c.id == key)).mappings().one()
            )
            if request["status"] != "running":
                raise ValueError("Request is already finalized")
            stamp = now().isoformat()
            if name == "create_note":
                if (
                    c.scalar(select(func.count()).select_from(NOTES))
                    >= self.config.max_records
                ):
                    raise RuntimeError("Note capacity reached")
                result = {"status": "created", "note_id": uuid.uuid4().hex, **values}
                c.execute(
                    insert(NOTES).values(
                        id=result["note_id"], created_at=stamp, **values
                    )
                )
            elif name == "list_notes":
                rows = c.execute(
                    select(NOTES).order_by(NOTES.c.created_at.desc()).limit(3)
                ).mappings()
                result = {
                    "notes": [
                        {
                            "note_id": r["id"],
                            "title": r["title"],
                            "body_preview": r["body"][:200],
                            "truncated": len(r["body"]) > 200,
                        }
                        for r in rows
                    ]
                }
            elif name == "create_reminder":
                if (
                    c.scalar(select(func.count()).select_from(REMINDERS))
                    >= self.config.max_records
                ):
                    raise RuntimeError("Reminder record capacity reached")
                pending = c.scalar(
                    select(func.count())
                    .select_from(REMINDERS)
                    .where(
                        REMINDERS.c.status.in_(
                            ["scheduled", "due", "claimed", "delivering"]
                        )
                    )
                )
                if pending >= self.config.max_pending_reminders:
                    raise RuntimeError("Pending reminder capacity reached")
                due = (
                    datetime.fromisoformat(request["created_at"])
                    + timedelta(seconds=values["delay_seconds"])
                    if "delay_seconds" in values
                    else datetime.fromisoformat(values["due_at"])
                )
                if due.tzinfo is None or not now() < due.astimezone(
                    timezone.utc
                ) <= now() + timedelta(days=366):
                    raise ValueError(
                        "Reminder must have an explicit timezone and a future time within one year"
                    )
                due = due.astimezone(timezone.utc)
                result = {
                    "status": "created",
                    "reminder_id": uuid.uuid4().hex,
                    "text": values["text"],
                    "due_at": due.isoformat(),
                }
                c.execute(
                    insert(REMINDERS).values(
                        id=result["reminder_id"],
                        text=values["text"],
                        due_at=due.isoformat(),
                        expires_at=(
                            due
                            + timedelta(seconds=self.config.notification_ttl_seconds)
                        ).isoformat(),
                        created_at=stamp,
                        status="scheduled",
                        attempts=0,
                    )
                )
            elif name == "list_reminders":
                rows = c.execute(
                    select(REMINDERS).order_by(REMINDERS.c.created_at.desc()).limit(5)
                ).mappings()
                result = {
                    "reminders": [
                        {
                            "reminder_id": r["id"],
                            "text": r["text"],
                            "due_at": r["due_at"],
                            "status": r["status"],
                        }
                        for r in rows
                    ]
                }
            else:
                row = (
                    c.execute(
                        select(REMINDERS).where(REMINDERS.c.id == values["reminder_id"])
                    )
                    .mappings()
                    .first()
                )
                status = (
                    "not_found"
                    if row is None
                    else (
                        "already_" + row["status"]
                        if row["status"] in {"delivered", "expired"}
                        else "cancelled"
                    )
                )
                result = {"status": status, **values, "prior_output_preserved": True}
                if row and status == "cancelled":
                    c.execute(
                        update(REMINDERS)
                        .where(REMINDERS.c.id == row["id"])
                        .values(status="cancelled")
                    )
            c.execute(
                insert(OPERATIONS).values(
                    id=operation,
                    request_id=key,
                    position=position,
                    fingerprint=digest,
                    name=name,
                    result=json.dumps(result, ensure_ascii=False),
                )
            )
        # If scheduling fails after the durable commit, the periodic reconciler
        # owns retry. Never report the committed note/reminder as rolled back.
        if name in {"create_reminder", "cancel_reminder"}:
            self.reconcile()
        if name == "create_reminder" and self.last_error:
            result = {
                **result,
                "scheduling_confirmed": False,
                "scheduler_error": self.last_error[:300],
            }
            with self._lock, self.db.begin() as c:
                c.execute(
                    update(OPERATIONS)
                    .where(OPERATIONS.c.id == operation)
                    .values(result=json.dumps(result, ensure_ascii=False))
                )
        return result

    def claim_due(self):
        with self._lock, self.db.begin() as c:
            self._open()
            stamp = now().isoformat()
            row = (
                c.execute(
                    select(REMINDERS)
                    .where(
                        REMINDERS.c.status == "due",
                        REMINDERS.c.expires_at > stamp,
                        (REMINDERS.c.next_attempt.is_(None))
                        | (REMINDERS.c.next_attempt <= stamp),
                    )
                    .order_by(REMINDERS.c.due_at)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if not row:
                return None
            attempt = uuid.uuid4().hex
            c.execute(
                update(REMINDERS)
                .where(REMINDERS.c.id == row["id"], REMINDERS.c.status == "due")
                .values(status="claimed", attempt=attempt, attempts=row["attempts"] + 1)
            )
            return {**row, "attempt": attempt}

    def begin_delivery(self, notice, channel):
        with self._lock, self.db.begin() as c:
            self._open()
            changed = c.execute(
                update(REMINDERS)
                .where(
                    REMINDERS.c.id == notice["id"],
                    REMINDERS.c.attempt == notice["attempt"],
                    REMINDERS.c.status == "claimed",
                    REMINDERS.c.expires_at > now().isoformat(),
                )
                .values(
                    status="delivering",
                    receipt=json.dumps({"kind": "started", "channel": channel}),
                )
            )
            if changed.rowcount != 1:
                raise RuntimeError("Reminder delivery is no longer allowed")

    def finish_delivery(
        self, notice, receipt=None, *, cancelled=False, error=None, release=False
    ):
        with self._lock, self.db.begin() as c:
            self._open()
            row = (
                c.execute(
                    select(REMINDERS).where(
                        REMINDERS.c.id == notice["id"],
                        REMINDERS.c.attempt == notice["attempt"],
                    )
                )
                .mappings()
                .first()
            )
            if not row or row["status"] not in {"claimed", "delivering"}:
                return
            if receipt and row["status"] == "delivering":
                status = "delivered"
            elif cancelled:
                status = "stopped"
            elif row["status"] == "delivering":
                status = "uncertain"
            elif release or row["attempts"] < 3:
                status = "due"
            else:
                status = "failed"
            c.execute(
                update(REMINDERS)
                .where(REMINDERS.c.id == notice["id"])
                .values(
                    status=status,
                    attempt=None,
                    attempts=max(0, row["attempts"] - 1)
                    if release
                    else row["attempts"],
                    next_attempt=(
                        now() + timedelta(seconds=0 if release else 5)
                    ).isoformat(),
                    receipt=json.dumps(
                        receipt or {"kind": status, "error": error}, ensure_ascii=False
                    ),
                )
            )

    def snapshot(self):
        # Monitoring reads must not hold the writer/notification mutex. The
        # application joins monitoring and Web readers before closing this DB.
        with self._lock:
            self._open()
            status = {
                "enabled": True,
                "scheduler_running": bool(self.scheduler.running),
                "error": self.last_error,
            }
        with self.db.connect() as c:
            return {
                **status,
                "notes": c.scalar(select(func.count()).select_from(NOTES)),
                "reminders": [
                    dict(r)
                    for r in c.execute(
                        select(REMINDERS)
                        .order_by(REMINDERS.c.created_at.desc())
                        .limit(10)
                    ).mappings()
                ],
            }
