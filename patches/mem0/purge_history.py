"""Project integration patch for mem0ai 2.2.1 SQLiteManager.

Upstream Memory.delete() retains old_memory/new_memory in its history database.
There is no per-memory history purge API. This version-pinned helper is the only
place that accesses the SDK's private database lock/connection. It erases rows
instead of retaining a body-bearing audit entry. Deletion tombstones belong to
the adapter's operation journal. This is logical erasure, not a claim to erase
old backups, SSD remanence, or third-party logs.

Reference: mem0/memory/storage.py at
b7ad69afda6b6ed030347c66d48a13e4de9dec08 (history/messages schemas).
"""

def purge_history(sdk, memory_ids: list[str], user_id: str) -> dict:
    manager = sdk.db
    if not hasattr(manager, "_lock") or not hasattr(manager, "connection"):
        raise RuntimeError("unsupported Mem0 history manager; purge not confirmed")
    with manager._lock:
        connection = manager.connection
        if connection is None:
            raise RuntimeError("Mem0 history database is closed")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(history)")}
        if not {"memory_id", "old_memory", "new_memory"} <= columns:
            raise RuntimeError("unsupported Mem0 history schema; purge not confirmed")
        # Version 2.2.1 escapes only the structural scope delimiters. Delimiter
        # boundaries prevent owner/owner2 confusion; LIKE metacharacters below
        # are separately escaped.
        encoded_user = str(user_id).replace("%", "%25").replace("&", "%26").replace("=", "%3D")
        pattern = f"%&user_id={encoded_user}&%"
        pattern = pattern.replace(encoded_user, encoded_user.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"))
        connection.execute("PRAGMA secure_delete=ON")
        try:
            connection.execute("BEGIN IMMEDIATE")
            removed = 0
            for memory_id in memory_ids:
                removed += connection.execute("DELETE FROM history WHERE memory_id=?", (memory_id,)).rowcount
            # Explicit infer=False writes do not create these messages. Purging
            # all retained SDK messages for the selected owner also prevents a
            # future candidate extractor from recreating a forgotten fact.
            messages = connection.execute("DELETE FROM messages WHERE ('&' || session_scope || '&') LIKE ? ESCAPE '\\'", (pattern,)).rowcount
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    return {"history_rows_removed": removed, "owner_message_rows_removed": messages}
