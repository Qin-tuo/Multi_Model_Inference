"""Actual local/native Qdrant and Mem0/SQLite; explicitly synthetic embeddings."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import unittest
import uuid

from qdrant_client import QdrantClient,models
from app.memory import Mem0Adapter
from app.memory_store import StoreBinding,check_transition
from app.records import Records
from deploy.memory.migrate import EmbeddingProfile,abort,migrate
from deploy.memory.prepare import BINARY
from test_mem0_adapter import FixtureEmbedding,actual_sdk
import test_memory_service as service_tests


@unittest.skipUnless(BINARY.exists(),'Pinned Qdrant binary must be prepared')
class MemoryMigrationTests(unittest.TestCase):
    setUpClass=classmethod(service_tests.NativeMemoryBindingTests.setUpClass.__func__)
    tearDownClass=classmethod(service_tests.NativeMemoryBindingTests.tearDownClass.__func__)

    def setUp(self):
        self.path=self.root/uuid.uuid4().hex
        self.path.mkdir()
        self.profile=EmbeddingProfile(FixtureEmbedding(),'f'*64,32,'FIXTURE')

    def seed(self,count=1):
        with Mem0Adapter(self.path,sdk=actual_sdk(self.path),embedding_scope='FIXTURE') as memory:
            return [memory.remember('owner','preference',f'key-{i}',f'合成事实{i}',source_id=f'source-{i}',request_id=f'save-{i}')['memory_id'] for i in range(count)]

    def run_migration(self,**kwargs):return migrate(self.path,self.service,self.profile,**kwargs)

    def target(self):return json.loads((self.path/'store.json').read_text())['collection']

    def test_ids_payloads_vectors_versions_history_and_owner_are_preserved(self):
        ids=self.seed(2)
        with Mem0Adapter(self.path,sdk=actual_sdk(self.path),embedding_scope='FIXTURE') as memory:
            memory.correct('owner',ids[0],'更正后合成事实',expected_version=1,source_id='correction',request_id='correct')
            guest=memory.remember('guest','preference','key-0','访客内容',source_id='guest')['memory_id']
            entity_id=str(uuid.uuid4())
            memory.sdk.entity_store.insert(vectors=[self.profile.embedder.embed('实体')],ids=[entity_id],payloads=[
                {'data':'实体','user_id':'owner','linked_memory_ids':ids}])
            before=memory.sdk.vector_store.client.scroll('assistant_memories',with_vectors=True)[0]
            history_before=memory.sdk.history(ids[0])
        result=self.run_migration()
        self.assertEqual(result['counts'],{'memories':3,'entities':1})
        with closing(self.service.client()) as client:
            after=client.retrieve(self.target(),[p.id for p in before],with_vectors=True)
            self.assertEqual({p.id:p.payload for p in before},{p.id:p.payload for p in after})
            self.assertEqual(next(p for p in after if p.id==guest).payload['user_id'],'guest')
        with closing(sqlite3.connect(self.path/'history.sqlite')) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM history WHERE memory_id=?',(ids[0],)).fetchone()[0],len(history_before))
        binding=StoreBinding(self.path,self.service,embedding_sha256='f'*64,dimensions=32)
        binding.close();check_transition(self.path)
        self.assertTrue(self.run_migration()['already_completed'])
        with self.assertRaisesRegex(RuntimeError,'bound to a native'):Mem0Adapter(self.path)

    def test_deletion_intents_and_descendants_do_not_resurrect(self):
        forgotten,record_deleted,kept=self.seed(3)
        records=self.path.parent/(self.path.name+'.records.sqlite')
        with Mem0Adapter(self.path,sdk=actual_sdk(self.path),embedding_scope='FIXTURE') as memory:
            child=memory.remember('owner','summary','derived','派生事实',source_id='summary',derived_from=(record_deleted,))['memory_id']
            old=memory.sdk.vector_store.client.retrieve('assistant_memories',[forgotten],with_vectors=True)[0]
            memory.forget('owner',forgotten,request_id='forget')
            # Labelled restored stale source point; current journal must win.
            memory.sdk.vector_store.client.upsert('assistant_memories',[models.PointStruct(id=old.id,vector=old.vector,payload=old.payload)])
            memory.sdk.db.add_history(forgotten,None,'应清除的旧快照正文','ADD')
            entities=[str(uuid.uuid4()) for _ in range(2)]
            memory.sdk.entity_store.insert(vectors=[self.profile.embedder.embed('实体')]*2,ids=entities,payloads=[
                {'data':'实体','user_id':'owner','linked_memory_ids':[forgotten,kept,record_deleted,child]},
                {'data':'实体','user_id':'owner','linked_memory_ids':[forgotten]}])
        with Records(records) as store:store.purge_memory('owner',record_deleted)
        result=self.run_migration(records=records)
        self.assertEqual(result['counts'],{'memories':1,'entities':1})
        self.assertEqual(result['deletion_barriers'],3)
        with closing(self.service.client()) as client:
            self.assertEqual([p.id for p in client.scroll(self.target())[0]],[kept])
            self.assertEqual(client.retrieve(self.target()+'_entities',entities)[0].payload['linked_memory_ids'],[kept])
        with closing(sqlite3.connect(self.path/'history.sqlite')) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM history WHERE memory_id IN (?,?,?)',(forgotten,record_deleted,child)).fetchone()[0],0)

    def test_faults_block_open_and_resume_without_duplicates(self):
        for phase in ('marked','batch','verified','ready'):
            with self.subTest(phase=phase):
                self.path=self.root/uuid.uuid4().hex;self.path.mkdir();self.seed(35)
                def fail(current):
                    if current==phase:raise OSError('LABELLED migration interruption')
                with self.assertRaisesRegex(OSError,'LABELLED'):self.run_migration(checkpoint=fail)
                with self.assertRaisesRegex(RuntimeError,'migration is incomplete'):Mem0Adapter(self.path)
                result=self.run_migration()
                self.assertEqual(result['counts']['memories'],35)
                self.assertEqual(json.loads((self.path/'store-transition.json').read_text())['status'],'COMPLETED')

    def test_writer_or_records_busy_is_rejected_before_transition(self):
        self.seed()
        with Mem0Adapter(self.path,sdk=actual_sdk(self.path),embedding_scope='FIXTURE'):
            with self.assertRaisesRegex(RuntimeError,'in use'):self.run_migration()
        self.assertFalse((self.path/'store-transition.json').exists())
        record_path=self.path.parent/(self.path.name+'.records.sqlite')
        with Records(record_path):
            with self.assertRaisesRegex(RuntimeError,'in use'):self.run_migration(records=record_path)
        self.assertFalse((self.path/'store-transition.json').exists())

    def test_model_mismatch_and_changed_source_are_rejected(self):
        ids=self.seed()
        class WrongEmbedding(FixtureEmbedding):
            def embed(self,*args,**kwargs):return super().embed('wrong')
        wrong=EmbeddingProfile(WrongEmbedding(),'f'*64,32,'FIXTURE')
        with self.assertRaisesRegex(RuntimeError,'embeddings differ'):migrate(self.path,self.service,wrong)
        with closing(QdrantClient(path=str(self.path/'qdrant'))) as client:
            client.set_payload('assistant_memories',{'source_id':'changed-after-interruption'},[ids[0]])
        with self.assertRaisesRegex(RuntimeError,'Source changed'):self.run_migration()

    def test_unrelated_target_metadata_is_never_overwritten(self):
        self.seed()
        def fail(phase):
            if phase=='batch':raise OSError('LABELLED')
        with self.assertRaises(OSError):self.run_migration(checkpoint=fail)
        with closing(self.service.client()) as client:
            client.update_collection(self.target(),metadata={'unrelated':True})
            before=client.get_collection(self.target()).config.metadata
        with self.assertRaisesRegex(RuntimeError,'metadata'):self.run_migration()
        with closing(self.service.client()) as client:self.assertEqual(client.get_collection(self.target()).config.metadata,before)

    def test_completed_retry_never_replays_retired_source_and_missing_binding_fails(self):
        ids=self.seed();self.run_migration()
        with closing(self.service.client()) as client:client.delete(self.target(),models.PointIdsList(points=ids),wait=True)
        self.assertTrue(self.run_migration()['already_completed'])
        with closing(self.service.client()) as client:self.assertEqual(client.count(self.target()).count,0)
        (self.path/'store.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'migration is incomplete'):Mem0Adapter(self.path)
        with self.assertRaisesRegex(RuntimeError,'migration is incomplete'):self.run_migration()

    def test_cross_owner_entities_are_rejected(self):
        ids=self.seed()
        with Mem0Adapter(self.path,sdk=actual_sdk(self.path),embedding_scope='FIXTURE') as memory:
            memory.sdk.entity_store.insert(vectors=[self.profile.embedder.embed('实体')],ids=[str(uuid.uuid4())],payloads=[
                {'data':'实体','user_id':'guest','linked_memory_ids':ids}])
        with self.assertRaisesRegex(RuntimeError,'crosses owners'):self.run_migration()

    def test_incomplete_abort_is_resumable_and_preserves_source(self):
        ids=self.seed()
        def fail(phase):
            if phase=='ready':raise OSError('LABELLED migration interruption')
        with self.assertRaises(OSError):self.run_migration(checkpoint=fail)
        collection=self.target()
        def fail_abort(phase):
            if phase=='binding_removed':raise OSError('LABELLED abort interruption')
        with self.assertRaises(OSError):abort(self.path,self.service,checkpoint=fail_abort)
        with self.assertRaisesRegex(RuntimeError,'migration is incomplete'):Mem0Adapter(self.path)
        self.assertEqual(abort(self.path,self.service)['status'],'ABORTED')
        with closing(self.service.client()) as client:self.assertFalse(client.collection_exists(collection))
        with Mem0Adapter(self.path,sdk=actual_sdk(self.path),embedding_scope='FIXTURE') as memory:
            self.assertEqual(memory.list_memories('owner')[0]['memory_id'],ids[0])
        self.run_migration()
        with self.assertRaisesRegex(RuntimeError,'cannot roll back'):abort(self.path,self.service)

    def test_abort_refuses_unrelated_target(self):
        self.seed()
        def fail(phase):
            if phase=='batch':raise OSError('LABELLED')
        with self.assertRaises(OSError):self.run_migration(checkpoint=fail)
        with closing(self.service.client()) as client:client.update_collection(self.target(),metadata={'unrelated':True})
        with self.assertRaisesRegex(RuntimeError,'unrelated'):abort(self.path,self.service)
        self.assertTrue((self.path/'store-transition.json').exists())
