"""Thin, local Mem0 integration. Mem0/Qdrant own memory content and embeddings.

SQLAlchemy owns a small operation journal, idempotency outcomes and deletion /
cancellation barriers; it is not a second fact or vector store. SDK writes and
the journal are separate commits. Failures remain PENDING until a retry can
reconcile them; capabilities explicitly report that cross-store atomicity is
unavailable. CAS is serialized through this adapter's one process writer. Direct
SDK clients must not bypass this boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import uuid
import re
from urllib.parse import urlsplit
from typing import Any

# This applies to this process before importing Mem0, not to any user's shell
# startup files or global configuration. Model endpoints are explicit below.
os.environ["MEM0_TELEMETRY"] = "false"


class MemoryAdapterError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code, self.retryable = code, retryable

    def as_dict(self):
        return {"code": self.code, "message": str(self), "retryable": self.retryable}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value, label, limit=8192):
    if not isinstance(value, str) or not value.strip() or "\x00" in value or len(value.encode()) > limit:
        raise MemoryAdapterError("INVALID_ARGUMENT", f"invalid {label}")
    return value.strip()


def build_mem0_config(data_dir: str | Path, bge_path: str | Path, *,
                      llama_base_url: str = "http://127.0.0.1:8088/v1",
                      llama_model: str = "assistant-vlm", dimensions: int = 512,
                      embedding_backend: str = "torch", qdrant_url: str | None = None,
                      qdrant_collection: str | None = None) -> dict:
    """Exact mem0ai 2.2.1 configuration. No model is downloaded by this helper.

    The ONNX path is explicit and requires the matching cached ONNX artifact and
    SentenceTransformer ONNX dependencies. Both modes force CPU embedding.
    Upstream HuggingFaceEmbedding passes model_kwargs to SentenceTransformer.
    """
    data = Path(data_dir).expanduser().resolve()
    model = Path(bge_path).expanduser().resolve()
    if embedding_backend not in {"torch", "onnx"}:
        raise MemoryAdapterError("INVALID_ARGUMENT", "embedding backend must be torch or onnx")
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or not 1 <= dimensions <= 4096:
        raise MemoryAdapterError("INVALID_ARGUMENT", "invalid embedding dimension")
    model_kwargs = {"device": "cpu", "local_files_only": True}
    if embedding_backend == "onnx":
        model_kwargs.update({"backend": "onnx", "model_kwargs": {
            "provider": "CPUExecutionProvider", "file_name": "onnx/model.onnx"}})
    vector_config = {"collection_name": "assistant_memories", "path": str(data / "qdrant"),
                     "embedding_model_dims": dimensions, "on_disk": True}
    if qdrant_url is not None:
        try:
            endpoint = urlsplit(qdrant_url)
            port = endpoint.port
        except (TypeError,ValueError):
            raise MemoryAdapterError('INVALID_ARGUMENT','Invalid Qdrant endpoint') from None
        if (endpoint.scheme != 'http' or endpoint.hostname not in {'127.0.0.1','::1'}
                or port is None or not 1024<=port<=65535 or endpoint.username or endpoint.password
                or endpoint.path not in {'','/'} or endpoint.query or endpoint.fragment
                or not isinstance(qdrant_collection,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,62}',qdrant_collection)):
            raise MemoryAdapterError('INVALID_ARGUMENT','Qdrant evaluation requires a numeric loopback HTTP endpoint and explicit collection')
        vector_config.pop('path')
        vector_config.update(host=endpoint.hostname,port=port,path=None,collection_name=qdrant_collection)
    elif qdrant_collection is not None:
        raise MemoryAdapterError('INVALID_ARGUMENT','Explicit Qdrant collection requires a loopback endpoint')
    return {
        "version": "v1.1",
        "history_db_path": str(data / "history.sqlite"),
        "llm": {"provider": "openai", "config": {
            "model": llama_model, "openai_base_url": llama_base_url, "api_key": "local",
            "temperature": 0, "max_tokens": 128}},
        "embedder": {"provider": "huggingface", "config": {
            "model": str(model), "embedding_dims": dimensions, "model_kwargs": model_kwargs}},
        "vector_store": {"provider": "qdrant", "config": vector_config},
    }


class Mem0Adapter:
    """A single process entry point for manually selected owner/guest profiles.

    Every read filters owner + namespace; every ID operation verifies ownership
    before touching the SDK. Explicit remember uses infer=False and no LLM call.
    A stable request_id is required for caller retries; omitting it generates a
    new operation. An embedding fixture may be injected solely in lifecycle
    tests and must declare embedding_scope='FIXTURE'.
    """

    def __init__(self, data_dir: str | Path, *, bge_path: str | Path | None = None,
                 llama_base_url: str = "http://127.0.0.1:8088/v1",
                 llama_model: str = "assistant-vlm", embedding_backend: str = "torch",
                 sdk=None, embedding_scope: str = "PRODUCTION"):
        from sqlalchemy import Column, Integer, MetaData, String, Table, Text, create_engine, event

        if importlib.metadata.version("mem0ai") != "2.2.1":
            raise MemoryAdapterError("UNAVAILABLE", "Mem0 adapter is pinned to mem0ai 2.2.1")
        if os.environ.get("OPENROUTER_API_KEY"):
            # Upstream checks this before its explicit OpenAI base_url. Preserve
            # the user's environment and reject accidental endpoint switching.
            raise MemoryAdapterError("INVALID_ARGUMENT", "OPENROUTER_API_KEY overrides Mem0's local LLM endpoint; use an isolated service environment")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._gate = threading.RLock()
        self._closed = False
        self.embedding_scope = embedding_scope
        self._lock_file = open(self.data_dir / "adapter.writer.lock", "a+")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            raise MemoryAdapterError("UNAVAILABLE", "memory adapter already has a writer") from exc
        self.engine = create_engine(f"sqlite:///{self.data_dir / 'operations.sqlite'}", connect_args={"timeout": 5.0})
        @event.listens_for(self.engine, "connect")
        def configure_sqlite(connection, _):
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA secure_delete=ON")
        metadata = MetaData()
        self.owners = Table("owner_generations", metadata,
                            Column("user_id", String, primary_key=True), Column("generation", Integer, nullable=False))
        self.operations = Table("operations", metadata,
                                Column("user_id", String, primary_key=True), Column("request_id", String, primary_key=True),
                                Column("operation", String, nullable=False), Column("fingerprint", String, nullable=False),
                                Column("status", String, nullable=False), Column("object_id", String),
                                Column("result_json", Text), Column("created_at", String, nullable=False), Column("last_error", String))
        self.barriers = Table("deletion_barriers", metadata,
                              Column("memory_id", String, primary_key=True), Column("user_id", String, nullable=False),
                              Column("generation", Integer, nullable=False), Column("status", String, nullable=False),
                              Column("created_at", String, nullable=False))
        self.cancelled_sources = Table("cancelled_sources", metadata,
                                       Column("user_id", String, primary_key=True), Column("source_id", String, primary_key=True))
        try:
            metadata.create_all(self.engine)
            if sdk is None:
                if bge_path is None or not Path(bge_path).expanduser().is_dir():
                    raise MemoryAdapterError("UNAVAILABLE", "cached local BGE directory is required")
                from mem0 import Memory
                config = build_mem0_config(self.data_dir, bge_path, llama_base_url=llama_base_url,
                                           llama_model=llama_model, embedding_backend=embedding_backend)
                if embedding_backend == "onnx":
                    import onnxruntime as ort
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 2
                    options.inter_op_num_threads = 1
                    config["embedder"]["config"]["model_kwargs"]["model_kwargs"]["session_options"] = options
                sdk = Memory.from_config(config)
                from app.adapters.embedding import BGEQueryAdapter
                sdk.embedding_model = BGEQueryAdapter(sdk.embedding_model)
            self.sdk = sdk
            from mem0.memory.telemetry import MEM0_TELEMETRY
            if MEM0_TELEMETRY:
                raise MemoryAdapterError("UNAVAILABLE", "Mem0 was imported before telemetry was disabled; restart service")
            self._recover_deletes()
        except BaseException:
            try:
                self._close_sdk()
            finally:
                self.engine.dispose()
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._gate:
            if self._closed:
                return
            self._closed = True
            # SDK uses a SQLite history manager plus a Qdrant client. Close each
            # explicitly so the local Qdrant path can be reopened after restart.
            try:
                self._close_sdk()
            finally:
                self.engine.dispose()
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()

    def _close_sdk(self):
        if not hasattr(self, "sdk"):
            return
        stores = [getattr(self.sdk, name, None) for name in ("vector_store", "_entity_store", "_telemetry_vector_store")]
        try:
            self.sdk.db.close()
        finally:
            seen = set()
            for store in stores:
                client = getattr(store, "client", None)
                if client is not None and id(client) not in seen:
                    seen.add(id(client))
                    client.close()

    def capabilities(self):
        return {"provider": "mem0ai", "version": "2.2.1", "vector_store":
                'qdrant_local' if self.sdk.vector_store.is_local else 'qdrant_server',
                "embedding_scope": self.embedding_scope, "explicit_memory_infer": False,
                "cas": "serialized_adapter_writer", "cross_store_atomicity": False,
                "operation_recovery": "retry_reconciliation", "owner_id_checks": True,
                "history_purge": "project_patch_for_2.2.1",
                "backup_erasure": "requires_current_deletion_journal",
                "semantic_quality_verified": False}

    def _check_open(self):
        if self._closed:
            raise MemoryAdapterError("UNAVAILABLE", "memory adapter is closed")

    def generation(self, user_id: str) -> int:
        from sqlalchemy import select
        user_id = _text(user_id, "user_id", 128)
        with self._gate:
            self._check_open()
            with self.engine.begin() as connection:
                value = connection.execute(select(self.owners.c.generation).where(self.owners.c.user_id == user_id)).scalar_one_or_none()
                if value is None:
                    connection.execute(self.owners.insert().values(user_id=user_id, generation=0))
                    return 0
                return value

    def _is_deleted(self, user_id, memory_id):
        from sqlalchemy import select
        with self.engine.connect() as connection:
            return connection.execute(select(self.barriers.c.memory_id).where(
                self.barriers.c.user_id == user_id, self.barriers.c.memory_id == memory_id)).first() is not None

    def _owned(self, user_id, memory_id, *, allow_deleted=False):
        memory_id = _text(memory_id, "memory_id", 128)
        if not allow_deleted and self._is_deleted(user_id, memory_id):
            raise MemoryAdapterError("NOT_FOUND", "memory was forgotten")
        record = self.sdk.get(memory_id)
        if record is None or record.get("user_id") != user_id:
            raise MemoryAdapterError("NOT_FOUND", "memory not found in selected profile")
        return record

    @staticmethod
    def _record(record):
        meta = record.get("metadata", {})
        return {"memory_id": record["id"], "user_id": record.get("user_id"),
                "content": record.get("memory", ""), "type": meta.get("memory_type", "fact"),
                "key": meta.get("key"), "namespace": meta.get("namespace", "personal"),
                "version": meta.get("version", 1), "generation": meta.get("generation", 0),
                "created_at": record.get("created_at"), "updated_at": record.get("updated_at"),
                "sources": [{"source_id": meta.get("source_id"), "evidence": meta.get("source_excerpt")}],
                "derived_from": meta.get("derived_from", []), "score": record.get("score")}

    def _find(self, user_id, namespace, extra=None, *, top_k=5):
        filters = {"user_id": user_id, "namespace": namespace, **(extra or {})}
        result = self.sdk.get_all(filters=filters, top_k=top_k)
        return [record for record in result.get("results", [])
                if record.get("user_id") == user_id and not self._is_deleted(user_id, record["id"])]

    def _iter_all(self, user_id, namespace, extra=None):
        """Use the mature Qdrant client's public cursor to avoid truncation."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        filters = {"user_id": user_id, "namespace": namespace, **(extra or {})}
        condition = Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))
                                 for key, value in filters.items()])
        offset = None
        while True:
            records, offset = self.sdk.vector_store.client.scroll(
                collection_name=self.sdk.collection_name, scroll_filter=condition,
                limit=256, offset=offset, with_payload=True, with_vectors=False)
            for row in records:
                memory_id = str(row.id)
                if not self._is_deleted(user_id, memory_id):
                    record = self._owned(user_id, memory_id)
                    yield record
            if offset is None:
                break

    def _start(self, user_id, request_id, operation, arguments):
        from sqlalchemy import select
        request_id = _text(request_id or uuid.uuid4().hex, "request_id", 256)
        fingerprint = hashlib.sha256(_dump(arguments).encode()).hexdigest()
        with self.engine.begin() as connection:
            row = connection.execute(select(self.operations).where(
                self.operations.c.user_id == user_id, self.operations.c.request_id == request_id)).mappings().first()
            if row:
                if row["operation"] != operation or row["fingerprint"] != fingerprint:
                    raise MemoryAdapterError("CONFLICT", "idempotency key reused with different operation or arguments")
                if row["status"] == "COMPLETED":
                    result = json.loads(row["result_json"])
                    memory_id = result.get("memory_id")
                    if memory_id and operation != "forget" and self._is_deleted(user_id, memory_id):
                        result = {"memory_id": memory_id, "deleted": True, "durable": True}
                    result["replayed"] = True
                    return request_id, result
                return request_id, None
            connection.execute(self.operations.insert().values(user_id=user_id, request_id=request_id,
                                                               operation=operation, fingerprint=fingerprint, status="PENDING", created_at=_now()))
        return request_id, None

    def _finish(self, user_id, request_id, result):
        with self.engine.begin() as connection:
            connection.execute(self.operations.update().where(self.operations.c.user_id == user_id,
                                                               self.operations.c.request_id == request_id).values(
                status="COMPLETED", object_id=result.get("memory_id"), result_json=_dump(result), last_error=None))
        return result

    def _pending_error(self, user_id, request_id, exc):
        with self.engine.begin() as connection:
            # Do not persist raw SDK exception text: some providers include
            # forgotten fact bodies or full request payloads in their errors.
            connection.execute(self.operations.update().where(self.operations.c.user_id == user_id,
                                                               self.operations.c.request_id == request_id).values(last_error=type(exc).__name__))
        raise MemoryAdapterError("STORAGE_ERROR", "Mem0 operation is pending; retry the same request_id to reconcile", retryable=True) from exc

    def _guard_source(self, user_id, source_id, expected_generation):
        from sqlalchemy import select
        generation = self.generation(user_id)
        if expected_generation is not None and expected_generation != generation:
            raise MemoryAdapterError("CANCELLED", "old memory generation cannot commit")
        with self.engine.connect() as connection:
            if connection.execute(select(self.cancelled_sources.c.source_id).where(
                self.cancelled_sources.c.user_id == user_id, self.cancelled_sources.c.source_id == source_id)).first():
                raise MemoryAdapterError("CANCELLED", "source turn was cancelled")
        return generation

    def cancel_source(self, user_id: str, source_id: str):
        from sqlalchemy import select
        user_id, source_id = _text(user_id, "user_id", 128), _text(source_id, "source_id", 256)
        with self._gate, self.engine.begin() as connection:
            self._check_open()
            found = connection.execute(select(self.cancelled_sources).where(
                self.cancelled_sources.c.user_id == user_id, self.cancelled_sources.c.source_id == source_id)).first()
            if found is None:
                connection.execute(self.cancelled_sources.insert().values(user_id=user_id, source_id=source_id))
        return {"cancelled": True, "source_id": source_id, "durable": True}

    def remember(self, user_id: str, memory_type: str, key: str, content: str, *, source_id: str,
                 source_excerpt: str | None = None, namespace: str = "personal", confirmed: bool = True,
                 request_id: str | None = None, expected_generation: int | None = None,
                 derived_from: tuple[str, ...] = ()) -> dict:
        user_id, key, content = _text(user_id, "user_id", 128), _text(key, "key", 512), _text(content, "content")
        source_id, namespace = _text(source_id, "source_id", 256), _text(namespace, "namespace", 128)
        excerpt = _text(source_excerpt if source_excerpt is not None else content, "source_excerpt")
        if memory_type not in {"fact", "preference", "event", "summary", "document"} or confirmed is not True:
            raise MemoryAdapterError("INVALID_ARGUMENT", "known memory type and explicit user confirmation required")
        if len(derived_from) > 20:
            raise MemoryAdapterError("INVALID_ARGUMENT", "at most 20 parent memories")
        arguments = {"type": memory_type, "key": key, "content": content, "source_id": source_id,
                     "source_excerpt": excerpt, "namespace": namespace, "expected_generation": expected_generation,
                     "derived_from": list(derived_from)}
        with self._gate:
            self._check_open()
            request_id, prior = self._start(user_id, request_id, "remember", arguments)
            if prior:
                return prior
            generation = self._guard_source(user_id, source_id, expected_generation)
            for parent_id in derived_from:
                self._owned(user_id, parent_id)
            recovered = self._find(user_id, namespace, {"operation_key": request_id}, top_k=2)
            if recovered:
                record = recovered[0]
                if record.get("memory") != content or record.get("metadata", {}).get("key") != key:
                    raise MemoryAdapterError("CONFLICT", "pending operation has conflicting persisted memory")
                try:
                    if not self.sdk.history(record["id"]):
                        self.sdk.db.add_history(record["id"], None, content, "ADD", created_at=record.get("created_at"))
                    return self._finish(user_id, request_id, {"memory_id": record["id"], "version": 1,
                                                             "generation": generation, "durable": True, "replayed": True,
                                                             "atomicity": "reconciled_non_atomic"})
                except Exception as exc:
                    self._pending_error(user_id, request_id, exc)
            if self._find(user_id, namespace, {"key": key, "memory_type": memory_type}, top_k=1):
                raise MemoryAdapterError("CONFLICT", "active fact key exists; use correct with expected_version")
            metadata = {"namespace": namespace, "memory_type": memory_type, "key": key, "version": 1,
                        "generation": generation, "source_id": source_id, "source_excerpt": excerpt,
                        "operation_key": request_id, "derived_from": list(derived_from)}
            try:
                result = self.sdk.add([{"role": "user", "content": content}], user_id=user_id, metadata=metadata, infer=False)
                memories = result.get("results", [])
                if len(memories) != 1 or memories[0].get("event") != "ADD":
                    raise RuntimeError("Mem0 did not confirm exactly one explicit fact")
                memory_id = memories[0]["id"]
                self._owned(user_id, memory_id)
                return self._finish(user_id, request_id, {"memory_id": memory_id, "version": 1, "generation": generation,
                                                         "durable": True, "replayed": False, "atomicity": "reconciled_non_atomic"})
            except Exception as exc:
                self._pending_error(user_id, request_id, exc)

    def search(self, user_id: str, query: str = "", *, key: str | None = None, namespace: str = "personal",
               memory_type: str | None = None, top_k: int = 5) -> dict:
        user_id, namespace = _text(user_id, "user_id", 128), _text(namespace, "namespace", 128)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
            raise MemoryAdapterError("INVALID_ARGUMENT", "top_k must be 1..5")
        with self._gate:
            self._check_open()
            filters = {"user_id": user_id, "namespace": namespace}
            if memory_type:
                filters["memory_type"] = memory_type
            if key is not None:
                records = self._find(user_id, namespace, {"key": _text(key, "key", 512), **({"memory_type": memory_type} if memory_type else {})}, top_k=top_k)
                method = "exact_metadata"
            else:
                query = _text(query, "query", 4096)
                result = self.sdk.search(query, filters=filters, top_k=top_k, rerank=False)
                records = [record for record in result.get("results", [])
                           if record.get("user_id") == user_id and not self._is_deleted(user_id, record["id"])]
                # SDK can quietly fall back when optional fastembed/BM25 is
                # absent. Report the actual configured backend after this call.
                bm25_ready = bool(getattr(self.sdk.vector_store, "_bm25_encoder", None)
                                  and getattr(self.sdk.vector_store, "_has_bm25_slot", False))
                method = "mem0_hybrid" if bm25_ready else "mem0_dense"
            return {"items": [self._record(record) for record in records], "retrieval_method": method,
                    "embedding_scope": self.embedding_scope, "generation": self.generation(user_id),
                    "degraded": method == "mem0_dense", "semantic_quality_verified": False}

    def list_memories(self, user_id: str, *, namespace: str = "personal", limit: int = 100) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise MemoryAdapterError("INVALID_ARGUMENT", "limit must be 1..1000")
        user_id, namespace = _text(user_id, "user_id", 128), _text(namespace, "namespace", 128)
        with self._gate:
            self._check_open()
            return [self._record(record) for record in self._find(user_id, namespace, top_k=limit)]

    def correct(self, user_id: str, memory_id: str, content: str, *, expected_version: int,
                source_id: str, source_excerpt: str | None = None, request_id: str | None = None,
                expected_generation: int | None = None) -> dict:
        user_id, content = _text(user_id, "user_id", 128), _text(content, "content")
        source_id = _text(source_id, "source_id", 256)
        excerpt = _text(source_excerpt if source_excerpt is not None else content, "source_excerpt")
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 1:
            raise MemoryAdapterError("INVALID_ARGUMENT", "expected_version must be positive integer")
        arguments = {"memory_id": memory_id, "content": content, "expected_version": expected_version,
                     "source_id": source_id, "source_excerpt": excerpt, "expected_generation": expected_generation}
        with self._gate:
            self._check_open()
            request_id, prior = self._start(user_id, request_id, "correct", arguments)
            if prior:
                return prior
            generation = self._guard_source(user_id, source_id, expected_generation)
            record = self._owned(user_id, memory_id)
            meta = record.get("metadata", {})
            current_version = meta.get("version", 1)
            if current_version == expected_version + 1 and meta.get("operation_key") == request_id and record.get("memory") == content:
                # A prior update committed its vector payload but journal commit
                # or SDK history failed. Repair the history then acknowledge.
                try:
                    history = self.sdk.history(memory_id)
                    if not any(entry.get("new_memory") == content and entry.get("event") == "UPDATE" for entry in history):
                        self.sdk.db.add_history(memory_id, None, content, "UPDATE", updated_at=record.get("updated_at"))
                    self._purge_descendants(user_id, memory_id, meta.get("namespace", "personal"))
                    return self._finish(user_id, request_id, {"memory_id": memory_id, "version": current_version,
                                                             "generation": self.generation(user_id), "durable": True,
                                                             "replayed": True, "atomicity": "reconciled_non_atomic"})
                except Exception as exc:
                    self._pending_error(user_id, request_id, exc)
            if current_version != expected_version:
                raise MemoryAdapterError("CONFLICT", "memory version changed")
            try:
                self.sdk.update(memory_id, text=content, metadata={"version": expected_version + 1,
                                                                  "generation": generation, "source_id": source_id,
                                                                  "source_excerpt": excerpt, "operation_key": request_id})
                self._purge_descendants(user_id, memory_id, meta.get("namespace", "personal"))
                return self._finish(user_id, request_id, {"memory_id": memory_id, "version": expected_version + 1,
                                                         "generation": self.generation(user_id), "durable": True,
                                                         "replayed": False, "atomicity": "reconciled_non_atomic"})
            except Exception as exc:
                self._pending_error(user_id, request_id, exc)

    def _descendants(self, user_id, memory_id, namespace):
        ids, pending = set(), [memory_id]
        while pending:
            parent = pending.pop()
            children = self._iter_all(user_id, namespace, {"derived_from": parent})
            for child in children:
                if child["id"] not in ids and child["id"] != memory_id:
                    ids.add(child["id"])
                    pending.append(child["id"])
        return sorted(ids)

    def _install_barriers(self, user_id, ids):
        from sqlalchemy import select
        generation = self.generation(user_id) + 1
        with self.engine.begin() as connection:
            connection.execute(self.owners.update().where(self.owners.c.user_id == user_id).values(generation=generation))
            for memory_id in ids:
                existing = connection.execute(select(self.barriers.c.memory_id).where(self.barriers.c.memory_id == memory_id)).first()
                if not existing:
                    connection.execute(self.barriers.insert().values(memory_id=memory_id, user_id=user_id,
                                                                   generation=generation, status="PENDING", created_at=_now()))
        return generation

    def _erase(self, user_id, ids):
        from patches.mem0.purge_history import purge_history
        for memory_id in ids:
            record = self.sdk.get(memory_id)
            if record is not None:
                if record.get("user_id") != user_id:
                    raise MemoryAdapterError("CONFLICT", "deletion barrier does not match SDK owner")
                self.sdk.delete(memory_id)
        result = purge_history(self.sdk, ids, user_id)
        with self.engine.begin() as connection:
            connection.execute(self.barriers.update().where(self.barriers.c.user_id == user_id,
                                                            self.barriers.c.memory_id.in_(ids)).values(status="COMPLETED"))
            # Journal result payloads never contain memory text. Fingerprints
            # remain hashes only; source evidence is owned by the SDK payload.
        return result

    def _purge_descendants(self, user_id, memory_id, namespace):
        ids = self._descendants(user_id, memory_id, namespace)
        if ids:
            self._install_barriers(user_id, ids)
            self._erase(user_id, ids)

    def forget(self, user_id: str, memory_id: str, *, expected_version: int | None = None,
               request_id: str | None = None) -> dict:
        from sqlalchemy import select
        user_id, memory_id = _text(user_id, "user_id", 128), _text(memory_id, "memory_id", 128)
        arguments = {"memory_id": memory_id, "expected_version": expected_version}
        with self._gate:
            self._check_open()
            request_id, prior = self._start(user_id, request_id, "forget", arguments)
            if prior:
                return prior
            already_deleted = self._is_deleted(user_id, memory_id)
            if not already_deleted:
                record = self._owned(user_id, memory_id)
                meta = record.get("metadata", {})
                if expected_version is not None and meta.get("version", 1) != expected_version:
                    raise MemoryAdapterError("CONFLICT", "memory version changed")
                ids = [memory_id] + self._descendants(user_id, memory_id, meta.get("namespace", "personal"))
                generation = self._install_barriers(user_id, ids)
            else:
                # Include any pending descendant barriers left by a failed
                # multi-record deletion, not just the root record.
                with self.engine.connect() as connection:
                    rows = connection.execute(select(self.barriers).where(self.barriers.c.user_id == user_id)).mappings().all()
                ids = [row["memory_id"] for row in rows if row["status"] == "PENDING" or row["memory_id"] == memory_id]
                generation = self.generation(user_id)
            try:
                erased = self._erase(user_id, ids)
                return self._finish(user_id, request_id, {"memory_id": memory_id, "deleted": True,
                                                         "deleted_ids": ids, "generation": generation,
                                                         "durable": True, "replayed": already_deleted,
                                                         "history_purged": True, "purge_counts": erased,
                                                         "atomicity": "reconciled_non_atomic"})
            except Exception as exc:
                self._pending_error(user_id, request_id, exc)

    def _recover_deletes(self):
        from sqlalchemy import select
        with self.engine.connect() as connection:
            rows = connection.execute(select(self.barriers)).mappings().all()
        owners = {}
        for row in rows:
            owners.setdefault(row["user_id"], []).append(row["memory_id"])
        # Reapply both pending and completed barriers. Restoring an old Qdrant
        # snapshot while retaining this journal cannot revive forgotten IDs.
        for user_id, ids in owners.items():
            self._erase(user_id, ids)

    def export(self, user_id: str, *, namespace: str = "personal") -> dict:
        user_id, namespace = _text(user_id, "user_id", 128), _text(namespace, "namespace", 128)
        with self._gate:
            self._check_open()
            return {"schema_version": 1, "user_id": user_id, "namespace": namespace,
                    "memories": [self._record(record) for record in self._iter_all(user_id, namespace)],
                    "exported_at": _now(), "complete": True}

    def operation_status(self, user_id: str, request_id: str) -> dict | None:
        from sqlalchemy import select
        with self._gate, self.engine.connect() as connection:
            self._check_open()
            row = connection.execute(select(self.operations).where(self.operations.c.user_id == user_id,
                                                                    self.operations.c.request_id == request_id)).mappings().first()
            return dict(row) if row else None
