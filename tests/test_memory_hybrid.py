"""Actual native BM25/provider/lifecycle; deterministic dense vectors, no LLM."""
from contextlib import closing
import json
import sqlite3
import unittest
import uuid

from qdrant_client import models
from app.memory import Mem0Adapter,build_mem0_config
from app.memory_profiles import WINDOWS,HYBRID,profile_of
from app.memory_store import StoreBinding,check_transition
from app.memory_lexical import valid_sparse
from deploy.data.backup import create
from deploy.data.restore import restore
from deploy.memory.migrate import EmbeddingProfile
from deploy.memory.prepare import BINARY
from deploy.memory.reindex import reindex,abort
from test_mem0_adapter import ForbiddenLLM
from test_memory_reindex import MatrixFixture
import test_memory_reindex as reindex_tests
import test_memory_service as service_tests


@unittest.skipUnless(BINARY.exists(),'Pinned native Qdrant required')
class HybridProviderTests(unittest.TestCase):
    setUpClass=classmethod(service_tests.NativeMemoryBindingTests.setUpClass.__func__)
    tearDownClass=classmethod(service_tests.NativeMemoryBindingTests.tearDownClass.__func__)

    def setUp(self):
        from mem0 import Memory
        from app.adapters.qdrant_hybrid import NativeHybridQdrant
        self.directory=self.root/uuid.uuid4().hex
        (self.directory/'binding').mkdir(parents=True)
        (self.directory/'adapter').mkdir()
        self.binding=StoreBinding(self.directory/'binding',self.service,embedding_sha256='e'*64,dimensions=32,profile=HYBRID)
        self.addCleanup(self.binding.close)
        config=build_mem0_config(self.directory/'adapter',self.directory/'fixture',dimensions=32,
            qdrant_url=self.service.url,qdrant_collection=self.binding.collection)
        config['vector_store']['config']['client']=self.binding.client
        config['embedder']={'provider':'openai','config':{'model':'FIXTURE','embedding_dims':32,'api_key':'unused',
            'openai_base_url':'http://127.0.0.1:1/v1'}}
        sdk=Memory.from_config(config);sdk.embedding_model=MatrixFixture();sdk.llm=ForbiddenLLM()
        sdk.vector_store=NativeHybridQdrant(sdk.vector_store)
        self.memory=Mem0Adapter(self.directory/'adapter',sdk=sdk,embedding_scope='FIXTURE')
        self.binding.ready()

    def tearDown(self):self.memory.close();self.binding.close()

    def test_canonical_update_owner_namespace_source_and_forget(self):
        m=self.memory
        original=m.remember('owner','fact','p','项目P12345的交付暗号是青竹七号。',source_id='s',request_id='save',source_date='2026-10-13')['memory_id']
        m.remember('guest','fact','g','项目P12345的交付暗号是访客机密。',source_id='g',request_id='g')
        m.remember('owner','fact','n','项目P12345的交付暗号是隔离档案。',source_id='n',request_id='n',namespace='other')
        result=m.search('owner','项目P12345的交付暗号',source_date='2026-10-13')
        self.assertEqual([x['memory_id'] for x in result['items']],[original])
        self.assertEqual(result['retrieval_method'],'qdrant_native_dbsf');self.assertFalse(result['degraded'])
        self.assertFalse(m.search('owner','项目P12345的交付暗号',source_date='2026-10-12')['items'])
        before=m.sdk.vector_store.client.retrieve(m.sdk.collection_name,[original],with_vectors=True)[0].vector['bm25']
        m.correct('owner',original,'项目P54321的交付暗号是海岚九号。',expected_version=1,source_id='changed',request_id='correct')
        after=m.sdk.vector_store.client.retrieve(m.sdk.collection_name,[original],with_vectors=True)[0].vector
        self.assertTrue(valid_sparse(after));self.assertNotEqual(before,after['bm25'])
        self.assertIn('海岚九号',m.search('owner','项目P54321的交付暗号')['items'][0]['content'])
        m.forget('owner',original,request_id='forget')
        self.assertFalse(m.search('owner','项目P54321的交付暗号')['items']);self.assertEqual(m.sdk.history(original),[])
        self.assertEqual(len(m.list_memories('guest')),1)

    def test_candidate_union_keeps_lexical_match_outside_dense_sixty(self):
        store=self.memory.sdk.vector_store
        vectors=[[[1.,0.]+[0.]*30] for _ in range(80)]+[[[0.,1.]+[0.]*30]]
        ids=[str(uuid.uuid4()) for _ in vectors]
        payloads=[{'data':f'合成档案P{i:05d}：设备记录。','user_id':'owner','namespace':'personal'} for i in range(80)]
        payloads.append({'data':'特殊项目P99999采用青竹交付暗号。','user_id':'owner','namespace':'personal'})
        store.insert(vectors,payloads,ids)
        condition=store._create_filter({'user_id':'owner','namespace':'personal'})
        dense=store.client.query_points(store.collection_name,query=vectors[0],query_filter=condition,limit=60).points
        self.assertNotIn(ids[-1],[str(x.id) for x in dense])
        merged=store.search('特殊项目P99999采用青竹交付暗号',vectors[0],top_k=60,filters={'user_id':'owner','namespace':'personal'})
        self.assertIn(ids[-1],[str(x.id) for x in merged])
        self.assertTrue(all(0<=x.score<=1 for x in merged))
        self.assertEqual([x.score for x in merged],sorted([x.score for x in merged],reverse=True))
        with self.assertRaises(ValueError):store.search('P99999',vectors[0])

    def test_profile_lock_and_incomplete_updates_rejected(self):
        binding=dict(self.binding.binding);self.assertEqual(binding['schema_version'],3)
        self.assertEqual(profile_of(binding),HYBRID)
        binding['retrieval_sha256']='0'*64
        with self.assertRaises(RuntimeError):profile_of(binding)
        with self.assertRaises(ValueError):self.memory.sdk.vector_store.update(str(uuid.uuid4()),payload={'data':'unindexed edit'})
        self.binding.client.update_collection(self.binding.collection,sparse_vectors_config={'bm25':models.SparseVectorParams(modifier=models.Modifier.NONE)})
        with self.assertRaises(RuntimeError):self.binding.validate()


@unittest.skipUnless(BINARY.exists(),'Pinned native Qdrant required')
class HybridReindexTests(unittest.TestCase):
    setUpClass=classmethod(service_tests.NativeMemoryBindingTests.setUpClass.__func__)
    tearDownClass=classmethod(service_tests.NativeMemoryBindingTests.tearDownClass.__func__)
    setUp=reindex_tests.ReindexTests.setUp

    def hybrid(self):return EmbeddingProfile(MatrixFixture(),'e'*64,32,'FIXTURE',HYBRID)

    def test_successive_upgrade_abort_resume_and_no_downgrade(self):
        reindex(self.app,self.profile);previous=json.loads((self.path/'reindex-transition.json').read_text())
        def crash(phase):
            if phase=='batch':raise RuntimeError('injected')
        with self.assertRaisesRegex(RuntimeError,'injected'):reindex(self.app,self.hybrid(),checkpoint=crash)
        with self.assertRaises(RuntimeError):check_transition(self.path)
        abort(self.app);self.assertEqual(json.loads((self.path/'reindex-transition.json').read_text()),previous)
        check_transition(self.path)
        with self.assertRaisesRegex(RuntimeError,'injected'):reindex(self.app,self.hybrid(),checkpoint=crash)
        self.assertEqual(reindex(self.app,self.hybrid())['status'],'PASS');check_transition(self.path)
        with closing(self.service.client()) as client:
            current=json.loads((self.path/'store.json').read_text())
            self.assertTrue(all(valid_sparse(x.vector) for x in client.scroll(current['collection'],with_vectors=True)[0]))
        self.assertTrue(reindex(self.app,self.hybrid())['already_completed'])
        with self.assertRaisesRegex(RuntimeError,'downgrade'):reindex(self.app,self.profile)

    def test_old_backup_and_deleted_sparse_content_cannot_revive(self):
        backup=self.root/uuid.uuid4().hex;target=self.root/uuid.uuid4().hex
        create(self.app,backup);reindex(self.app,self.profile);reindex(self.app,self.hybrid())
        with sqlite3.connect(self.path/'operations.sqlite') as db:
            db.execute("INSERT INTO deletion_barriers VALUES (?,?,1,'PENDING','FIXTURE')",(self.drop,'owner'))
        self.assertEqual(restore(backup,self.app,target,self.service,self.hybrid())['status'],'PASS')
        binding=json.loads((target/'memory/store.json').read_text())
        with closing(self.service.client()) as client:
            values=client.retrieve(binding['collection'],[self.keep,self.drop,self.guest],with_vectors=True)
            self.assertEqual({str(x.id) for x in values},{self.keep,self.guest})
            self.assertTrue(all(valid_sparse(x.vector) for x in values))

    def test_same_profile_restore_rebuilds_corrupt_lexical_vectors(self):
        reindex(self.app,self.hybrid());binding=json.loads((self.path/'store.json').read_text())
        with closing(self.service.client()) as client:
            proper=client.retrieve(binding['collection'],[self.keep],with_vectors=True)[0].vector['bm25']
            client.update_vectors(binding['collection'],[models.PointVectors(id=self.keep,vector={'bm25':models.SparseVector(indices=[123],values=[999.])})],wait=True)
        backup=self.root/uuid.uuid4().hex;target=self.root/uuid.uuid4().hex;create(self.app,backup)
        self.assertEqual(restore(backup,self.app,target,self.service,self.hybrid())['status'],'PASS')
        restored=json.loads((target/'memory/store.json').read_text())
        with closing(self.service.client()) as client:
            self.assertEqual(client.retrieve(restored['collection'],[self.keep],with_vectors=True)[0].vector['bm25'],proper)
