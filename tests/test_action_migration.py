# SPDX-License-Identifier: Apache-2.0
"""Naming changes must preserve notes, receipts and scheduled reminders."""

from dataclasses import replace
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import create_engine, event, inspect, select

from app.actions.migrations import LEGACY_TABLES, migrate_legacy_schema
from app.actions.store import ActionStore, NOTES
from app.config import ActionsConfig


def test_existing_actions_and_jobs_survive_schema_and_path_migration(tmp_path):
    previous = tmp_path / "previous.sqlite"
    config = replace(ActionsConfig(), database=str(previous))
    store = ActionStore(config)
    try:
        store.begin_request("0123456789abcdef0123456789abcdef", "input")
        note = store.execute(
            "0123456789abcdef0123456789abcdef",
            0,
            "create_note",
            {"title": "保留", "body": "迁移前内容"},
        )
        reminder = store.execute(
            "0123456789abcdef0123456789abcdef",
            1,
            "create_reminder",
            {
                "text": "迁移后提醒",
                "due_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            },
        )
        store.finish_request("0123456789abcdef0123456789abcdef", "complete")
    finally:
        store.close()
    with closing(sqlite3.connect(previous)) as db, db:
        for old, new in LEGACY_TABLES.items():
            db.execute(f'ALTER TABLE "{new}" RENAME TO "{old}"')
        db.execute("DROP INDEX ix_assistant_action_operations_request_id")
        db.execute(
            "CREATE INDEX ix_reachy_action_operations_request_id ON reachy_action_operations (request_id)"
        )
    current = tmp_path / "current.sqlite"
    previous.rename(current)
    config = replace(config, database=str(current))
    for _ in range(2):
        reopened = ActionStore(config)
        try:
            snapshot = reopened.snapshot()
            assert snapshot["notes"] == 1
            with reopened.db.connect() as connection:
                saved_note = connection.execute(select(NOTES)).mappings().one()
                assert saved_note["id"] == note["note_id"]
                assert saved_note["body"] == "迁移前内容"
            assert snapshot["reminders"][0]["id"] == reminder["reminder_id"]
            assert [
                row["result"]
                for row in reopened.receipts("0123456789abcdef0123456789abcdef")
            ] == [note, reminder]
            jobs = reopened.scheduler.get_jobs(jobstore="default")
            assert len(jobs) == 1 and Path(jobs[0].args[0]) == current
            assert (
                not set(inspect(reopened.db).get_table_names()) & LEGACY_TABLES.keys()
            )
        finally:
            reopened.close()


@pytest.mark.parametrize("extra", ["assistant_notes", "unrelated_data"])
def test_migration_rejects_ambiguous_or_unrelated_database(tmp_path, extra):
    db = create_engine("sqlite:///" + str(tmp_path / "actions.sqlite"))
    try:
        with db.begin() as c:
            c.exec_driver_sql("CREATE TABLE reachy_notes (id TEXT)")
            c.exec_driver_sql(f'CREATE TABLE "{extra}" (id TEXT)')
        before = set(inspect(db).get_table_names())
        with pytest.raises(ValueError, match="refusing migration"):
            migrate_legacy_schema(db)
        assert set(inspect(db).get_table_names()) == before
    finally:
        db.dispose()


def test_failed_migration_rolls_back_all_table_renames(tmp_path):
    db = create_engine("sqlite:///" + str(tmp_path / "actions.sqlite"))
    try:
        with db.begin() as c:
            for name in LEGACY_TABLES:
                c.exec_driver_sql(f'CREATE TABLE "{name}" (id TEXT)')

        def fail_second_rename(conn, cursor, statement, parameters, context, many):
            if statement.startswith('ALTER TABLE "reachy_action_operations"'):
                raise RuntimeError("injected DDL failure")

        event.listen(db, "before_cursor_execute", fail_second_rename)
        with pytest.raises(RuntimeError, match="injected"):
            migrate_legacy_schema(db)
        assert set(inspect(db).get_table_names()) == set(LEGACY_TABLES)
    finally:
        db.dispose()
