"""Real Mem0 2.2.1 + local Qdrant lifecycle tests, synthetic embeddings only.

FIXTURE vectors do not prove BGE quality, semantic recall, 10k-record latency,
or the full JT-10/JT-11 gates. No LLM/network calls are allowed in these tests.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import uuid

from app.memory import Mem0Adapter, MemoryAdapterError, build_mem0_config


HAVE_SDK = all(importlib.util.find_spec(name) for name in ("mem0", "sqlalchemy", "qdrant_client"))


class FixtureEmbedding:
    """Deterministic contract vector, deliberately not a semantic model."""

    def embed(self, text, memory_action=None):
        digest = hashlib.sha256(text.encode()).digest()
        vector = [float(value - 127) for value in digest]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def embed_batch(self, texts, memory_action="add"):
        return [self.embed(text, memory_action) for text in texts]


class ForbiddenLLM:
    def generate_response(self, *_, **__):
        raise AssertionError("explicit lifecycle operations must not call an LLM")


def actual_sdk(data_dir):
    """Construct the real SDK, swapping just the embedding inference provider."""
    from mem0 import Memory
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    config = build_mem0_config(data_dir, data_dir / "fixture-model", dimensions=32)
    # OpenAI provider construction itself makes no request. Before any operation
    # replace it with deterministic embeddings; any unexpected LLM call fails.
    config["embedder"] = {"provider": "openai", "config": {
        "model": "FIXTURE-no-semantic-model", "embedding_dims": 32,
        "api_key": "unused-fixture", "openai_base_url": "http://127.0.0.1:1/v1"}}
    sdk = Memory.from_config(config)
    sdk.embedding_model = FixtureEmbedding()
    sdk.llm = ForbiddenLLM()
    return sdk


class ConfigTests(unittest.TestCase):
    def test_server_evaluation_requires_explicit_numeric_loopback_and_collection(self):
        config=build_mem0_config('/tmp/config-test','/tmp/cached-bge',qdrant_url='http://127.0.0.1:16333',
                                 qdrant_collection='capacity_10000')['vector_store']['config']
        self.assertEqual((config['host'],config['port'],config['path'],config['collection_name']),
                         ('127.0.0.1',16333,None,'capacity_10000'))
        for url,collection in [('http://example.com:6333','capacity_1'),('http://127.0.0.1:invalid','capacity_1'),
                ('http://127.0.0.1:6333/path','capacity_1'),('http://user:password@127.0.0.1:6333','capacity_1'),
                ('http://127.0.0.1:6333',None),('http://127.0.0.1:6333','../collection'),(None,'capacity_1')]:
            with self.subTest(url=url),self.assertRaises(MemoryAdapterError):
                build_mem0_config('/tmp/config-test','/tmp/cached-bge',qdrant_url=url,qdrant_collection=collection)

    def test_config_pins_local_llm_cpu_embedding_and_persistent_paths(self):
        config = build_mem0_config("/tmp/config-test", "/tmp/cached-bge", embedding_backend="onnx")
        self.assertEqual("http://127.0.0.1:8088/v1", config["llm"]["config"]["openai_base_url"])
        self.assertEqual("assistant-vlm", config["llm"]["config"]["model"])
        self.assertEqual("huggingface", config["embedder"]["provider"])
        kwargs = config["embedder"]["config"]["model_kwargs"]
        self.assertEqual("cpu", kwargs["device"])
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual("onnx", kwargs["backend"])
        self.assertEqual("CPUExecutionProvider", kwargs["model_kwargs"]["provider"])
        self.assertEqual(512, config["vector_store"]["config"]["embedding_model_dims"])
        self.assertTrue(config["vector_store"]["config"]["on_disk"])


@unittest.skipUnless(HAVE_SDK, "real Mem0/Qdrant/SQLAlchemy dependencies are not installed")
class Mem0LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.adapter = Mem0Adapter(self.path, sdk=actual_sdk(self.path), embedding_scope="FIXTURE")

    def tearDown(self):
        self.adapter.close()
        self.tmp.cleanup()

    def remember(self, key="reply_style", content="简短", request_id="save", **kwargs):
        return self.adapter.remember("owner", "preference", key, content, source_id="turn-1", request_id=request_id, **kwargs)

    def assertCode(self, code, callback):
        with self.assertRaises(MemoryAdapterError) as raised:
            callback()
        self.assertEqual(code, raised.exception.code)

    def restart(self):
        self.adapter.close()
        self.adapter = Mem0Adapter(self.path, sdk=actual_sdk(self.path), embedding_scope="FIXTURE")

    def test_real_sdk_explicit_save_and_new_instance_recall(self):
        result = self.remember()
        self.assertTrue(result["durable"])
        self.assertEqual("COMPLETED", self.adapter.operation_status("owner", "save")["status"])
        self.restart()
        found = self.adapter.search("owner", key="reply_style")
        self.assertEqual("exact_metadata", found["retrieval_method"])
        self.assertEqual("FIXTURE", found["embedding_scope"])
        self.assertEqual(result["memory_id"], found["items"][0]["memory_id"])
        self.assertEqual("turn-1", found["items"][0]["sources"][0]["source_id"])

    def test_idempotency_survives_restart_and_argument_conflict(self):
        first = self.remember()
        self.restart()
        second = self.remember()
        self.assertTrue(second["replayed"])
        self.assertEqual(first["memory_id"], second["memory_id"])
        self.assertEqual(1, len(self.adapter.list_memories("owner")))
        self.assertCode("CONFLICT", lambda: self.remember(content="详细"))

    def test_owner_check_precedes_every_id_update_or_delete(self):
        memory_id = self.remember()["memory_id"]
        self.assertEqual([], self.adapter.search("guest", key="reply_style")["items"])
        self.assertEqual([], self.adapter.list_memories("guest"))
        self.assertCode("NOT_FOUND", lambda: self.adapter.correct("guest", memory_id, "bad", expected_version=1, source_id="g"))
        self.assertCode("NOT_FOUND", lambda: self.adapter.forget("guest", memory_id))
        self.assertEqual("简短", self.adapter.search("owner", key="reply_style")["items"][0]["content"])

    def test_serialized_adapter_cas_has_one_winner(self):
        memory_id = self.remember()["memory_id"]
        def correction(number):
            try:
                return self.adapter.correct("owner", memory_id, f"详细{number}", expected_version=1,
                                            source_id=f"c-{number}", request_id=f"c-{number}")
            except MemoryAdapterError as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(correction, [1, 2]))
        self.assertEqual(1, sum(isinstance(value, dict) for value in results))
        self.assertEqual(1, results.count("CONFLICT"))
        self.assertEqual(2, self.adapter.search("owner", key="reply_style")["items"][0]["version"])
        self.assertFalse(self.adapter.capabilities()["cross_store_atomicity"])

    def test_real_update_replay_preserves_new_version(self):
        memory_id = self.remember()["memory_id"]
        first = self.adapter.correct("owner", memory_id, "详细", expected_version=1, source_id="c", request_id="correct")
        second = self.adapter.correct("owner", memory_id, "详细", expected_version=1, source_id="c", request_id="correct")
        self.assertEqual(2, first["version"])
        self.assertTrue(second["replayed"])
        self.assertEqual(2, self.adapter.search("owner", key="reply_style")["items"][0]["version"])

    def test_sdk_delete_alone_retains_body_but_adapter_forget_purges_it(self):
        memory_id = self.remember(content="合成遗忘测试内容")["memory_id"]
        self.assertTrue(self.adapter.sdk.history(memory_id))
        result = self.adapter.forget("owner", memory_id, request_id="forget")
        self.assertTrue(result["history_purged"])
        self.assertIsNone(self.adapter.sdk.get(memory_id))
        self.assertEqual([], self.adapter.sdk.history(memory_id))
        self.assertEqual([], self.adapter.search("owner", key="reply_style")["items"])
        self.restart()
        self.assertEqual([], self.adapter.list_memories("owner"))
        replay = self.remember(content="合成遗忘测试内容")
        self.assertTrue(replay["deleted"])
        self.assertEqual([], self.adapter.list_memories("owner"))

    def test_forget_erases_derived_summary(self):
        parent = self.remember()["memory_id"]
        summary = self.adapter.remember("owner", "summary", "summary", "我喜欢简短", source_id="summary", request_id="summary", derived_from=(parent,))
        result = self.adapter.forget("owner", parent, request_id="forget")
        self.assertEqual({parent, summary["memory_id"]}, set(result["deleted_ids"]))
        self.assertEqual([], self.adapter.sdk.history(summary["memory_id"]))
        self.assertEqual([], self.adapter.list_memories("owner"))

    def test_entity_cleanup_after_restart_preserves_other_owners_and_live_links(self):
        forgotten=self.remember()["memory_id"]
        kept=self.remember(key='other',content='保留内容',request_id='keep')['memory_id']
        entity=self.adapter.sdk.entity_store
        ids=[str(uuid.uuid4()) for _ in range(3)]
        entity.insert(vectors=[FixtureEmbedding().embed('entity')]*3,ids=ids,payloads=[
            {'data':'删除的实体','user_id':'owner','linked_memory_ids':[forgotten]},
            {'data':'共享实体','user_id':'owner','linked_memory_ids':[forgotten,kept]},
            {'data':'其他档案实体','user_id':'guest','linked_memory_ids':[forgotten]}])
        self.restart()
        self.assertIsNone(self.adapter.sdk._entity_store)
        self.adapter.forget('owner',forgotten,request_id='forget-entity')
        rows=self.adapter.sdk.vector_store.client.retrieve('assistant_memories_entities',ids)
        by_id={str(row.id):row.payload for row in rows}
        self.assertNotIn(ids[0],by_id)
        self.assertEqual(by_id[ids[1]]['linked_memory_ids'],[kept])
        self.assertEqual(by_id[ids[2]]['linked_memory_ids'],[forgotten])

    def test_entity_cleanup_failure_stays_pending_and_recovers_on_reopen(self):
        memory_id=self.remember()['memory_id']
        entity=self.adapter.sdk.entity_store
        entity_id=str(uuid.uuid4())
        entity.insert(vectors=[FixtureEmbedding().embed('entity')],ids=[entity_id],payloads=[
            {'data':'合成遗忘实体','user_id':'owner','linked_memory_ids':[memory_id]}])
        self.restart()
        client=self.adapter.sdk.vector_store.client;original=client.delete
        def fail_entity(collection_name,*args,**kwargs):
            if collection_name.endswith('_entities'):raise OSError('LABELLED entity erasure failure')
            return original(collection_name,*args,**kwargs)
        client.delete=fail_entity
        self.assertCode('STORAGE_ERROR',lambda:self.adapter.forget('owner',memory_id,request_id='forget-entity'))
        self.assertEqual(self.adapter.operation_status('owner','forget-entity')['status'],'PENDING')
        client.delete=original
        self.restart()
        self.assertEqual(self.adapter.sdk.vector_store.client.retrieve('assistant_memories_entities',[entity_id]),[])
        self.assertTrue(self.adapter.forget('owner',memory_id,request_id='forget-entity')['durable'])

    def test_correction_invalidates_old_derived_summary(self):
        parent = self.remember()["memory_id"]
        child = self.adapter.remember("owner", "summary", "summary", "我喜欢简短", source_id="summary", request_id="summary", derived_from=(parent,))
        self.adapter.correct("owner", parent, "详细", expected_version=1, source_id="c", request_id="correct")
        self.assertIsNone(self.adapter.sdk.get(child["memory_id"]))
        self.assertEqual("详细", self.adapter.search("owner", key="reply_style")["items"][0]["content"])

    def test_old_background_generation_and_cancelled_turn_cannot_write(self):
        parent = self.remember()["memory_id"]
        self.adapter.forget("owner", parent)
        self.assertCode("CANCELLED", lambda: self.remember(key="late", request_id="late", expected_generation=0))
        self.adapter.cancel_source("owner", "turn-1")
        self.assertCode("CANCELLED", lambda: self.remember(key="cancelled", request_id="cancelled"))
        self.assertEqual([], self.adapter.list_memories("owner"))

    def test_partial_sdk_add_failure_is_pending_and_retry_reconciles(self):
        original = self.adapter.sdk.db.add_history
        def fail(*_, **__):
            raise RuntimeError("injected post-vector history failure")
        self.adapter.sdk.db.add_history = fail
        self.assertCode("STORAGE_ERROR", lambda: self.remember())
        self.assertEqual("PENDING", self.adapter.operation_status("owner", "save")["status"])
        self.adapter.sdk.db.add_history = original
        recovered = self.remember()
        self.assertTrue(recovered["replayed"])
        self.assertTrue(self.adapter.sdk.history(recovered["memory_id"]))
        self.assertEqual(1, len(self.adapter.list_memories("owner")))
        self.assertEqual("COMPLETED", self.adapter.operation_status("owner", "save")["status"])

    def test_delete_failure_preserves_barrier_and_restarts_to_finish(self):
        memory_id = self.remember()["memory_id"]
        original = self.adapter.sdk.delete
        def fail(*_, **__):
            raise RuntimeError("injected unavailable vector store")
        self.adapter.sdk.delete = fail
        self.assertCode("STORAGE_ERROR", lambda: self.adapter.forget("owner", memory_id, request_id="forget"))
        self.assertEqual("PENDING", self.adapter.operation_status("owner", "forget")["status"])
        self.assertIsNotNone(self.adapter.sdk.get(memory_id))
        self.assertEqual([], self.adapter.search("owner", key="reply_style")["items"])
        self.adapter.sdk.delete = original
        self.restart()
        self.assertIsNone(self.adapter.sdk.get(memory_id))
        self.assertEqual([], self.adapter.sdk.history(memory_id))
        retry = self.adapter.forget("owner", memory_id, request_id="forget")
        self.assertTrue(retry["history_purged"])
        self.assertEqual("COMPLETED", self.adapter.operation_status("owner", "forget")["status"])

    def test_completed_barrier_filters_restored_old_vector(self):
        memory_id = self.remember()["memory_id"]
        raw = self.adapter.sdk.vector_store.get(vector_id=memory_id)
        self.adapter.forget("owner", memory_id)
        # Simulate a stale Qdrant restore. The current journal must be retained.
        self.adapter.sdk.vector_store.insert(vectors=[FixtureEmbedding().embed("简短")],
                                             ids=[memory_id], payloads=[raw.payload])
        self.assertEqual([], self.adapter.search("owner", key="reply_style")["items"])
        self.restart()
        self.assertIsNone(self.adapter.sdk.get(memory_id))

    def test_purge_owner_message_scope_does_not_match_other_owner(self):
        self.adapter.sdk.db.save_messages([{"role": "user", "content": "owner text"}], "user_id=owner")
        self.adapter.sdk.db.save_messages([{"role": "user", "content": "owner2 text"}], "user_id=owner2")
        memory_id = self.remember()["memory_id"]
        self.adapter.forget("owner", memory_id)
        self.assertEqual([], self.adapter.sdk.db.get_last_messages("user_id=owner"))
        self.assertEqual(1, len(self.adapter.sdk.db.get_last_messages("user_id=owner2")))

    def test_purge_special_owner_scope_uses_upstream_delimiter_escaping(self):
        from mem0.memory.main import _build_session_scope
        user_id = "甲&乙=3%/x_y"
        scope = _build_session_scope({"user_id": user_id})
        other_scope = _build_session_scope({"user_id": user_id + "2"})
        self.adapter.sdk.db.save_messages([{"role": "user", "content": "合成待忘内容"}], scope)
        self.adapter.sdk.db.save_messages([{"role": "user", "content": "另一个owner"}], other_scope)
        memory = self.adapter.remember(user_id, "fact", "fact", "合成待忘内容", source_id="t")
        self.adapter.forget(user_id, memory["memory_id"])
        self.assertEqual([], self.adapter.sdk.db.get_last_messages(scope))
        self.assertEqual(1, len(self.adapter.sdk.db.get_last_messages(other_scope)))

    def test_namespace_and_fixture_quality_are_explicit(self):
        self.remember()
        self.assertEqual([], self.adapter.search("owner", key="reply_style", namespace="documents")["items"])
        self.assertEqual("FIXTURE", self.adapter.capabilities()["embedding_scope"])
        self.assertFalse(self.adapter.capabilities()["semantic_quality_verified"])
        self.assertCode("INVALID_ARGUMENT", lambda: self.remember(key="bad", request_id="bad", confirmed=False))
        exported = self.adapter.export("owner")
        self.assertTrue(exported["complete"])
        self.assertEqual(1, len(exported["memories"]))

    def test_sdk_query_reports_actual_dense_fallback_and_fixture_scope(self):
        self.remember()
        result = self.adapter.search("owner", "简短")
        self.assertEqual("FIXTURE", result["embedding_scope"])
        self.assertFalse(result["semantic_quality_verified"])
        if self.adapter.sdk.vector_store._bm25_encoder is False:
            self.assertEqual("mem0_dense", result["retrieval_method"])
            self.assertTrue(result["degraded"])
        self.assertEqual(1, len(result["items"]))

    def test_export_uses_real_qdrant_cursor_across_pages(self):
        # Seed via the real SDK backend to exercise pagination without 257
        # journal operations. These are still clearly synthetic contract data.
        ids = [str(__import__("uuid").uuid4()) for _ in range(257)]
        vector = FixtureEmbedding().embed("fixture")
        self.adapter.sdk.vector_store.insert(vectors=[vector] * len(ids), ids=ids,
                                             payloads=[{"data": f"FIXTURE {number}", "user_id": "owner",
                                                        "namespace": "personal", "key": f"key-{number}",
                                                        "version": 1} for number in range(len(ids))])
        exported = self.adapter.export("owner")
        self.assertTrue(exported["complete"])
        self.assertEqual(set(ids), {item["memory_id"] for item in exported["memories"]})
        self.assertEqual([], self.adapter.export("guest")["memories"])


if __name__ == "__main__":
    unittest.main()
