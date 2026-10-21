# SPDX-License-Identifier: Apache-2.0
"""在已有动作数据库上迁移固定旧表名，保留业务记录并拒绝混入无关数据表。"""

from sqlalchemy import inspect


LEGACY_TABLES = {
    "reachy_action_requests": "assistant_action_requests",
    "reachy_action_operations": "assistant_action_operations",
    "reachy_notes": "assistant_notes",
    "reachy_reminders": "assistant_reminders",
}


def migrate_legacy_schema(db):
    """在 ActionStore 的数据库所有权锁内，把已知旧表和索引改为当前名称。

    使用显式 SQLite 事务保证重命名整体提交；旧新表混用或包含无关表时拒绝迁移。
    """
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
