# SPDX-License-Identifier: Apache-2.0
"""One-time naming migration for this application's existing action database."""

from sqlalchemy import inspect


LEGACY_TABLES = {
    "reachy_action_requests": "assistant_action_requests",
    "reachy_action_operations": "assistant_action_operations",
    "reachy_notes": "assistant_notes",
    "reachy_reminders": "assistant_reminders",
}


def migrate_legacy_schema(db):
    """Run under ActionStore's owner lock; preserve rows and reject mixed stores."""
    with db.connect() as connection:
        # SQLite DDL needs an explicit transaction for all renames to roll back
        # together. Fixed identifiers below never include caller-supplied text.
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        tables = set(inspect(connection).get_table_names())
        legacy = tables & LEGACY_TABLES.keys()
        if legacy:
            if tables & set(LEGACY_TABLES.values()):
                raise ValueError(
                    "Mixed legacy/current action tables; refusing migration"
                )
            if tables - LEGACY_TABLES.keys() - {"apscheduler_jobs"}:
                raise ValueError(
                    "Database contains unrelated tables; refusing migration"
                )
            for old, new in LEGACY_TABLES.items():
                if old in legacy:
                    connection.exec_driver_sql(f'ALTER TABLE "{old}" RENAME TO "{new}"')
            indexes = {
                row[0]
                for row in connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
            if "ix_reachy_action_operations_request_id" in indexes:
                connection.exec_driver_sql(
                    "DROP INDEX ix_reachy_action_operations_request_id"
                )
                connection.exec_driver_sql(
                    "CREATE INDEX ix_assistant_action_operations_request_id "
                    "ON assistant_action_operations (request_id)"
                )
        connection.commit()
