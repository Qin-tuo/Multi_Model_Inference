"""Native reindex transitions, with explicitly labelled deterministic vectors."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import unittest
import uuid

from qdrant_client import models
from app.memory import Mem0Adapter
from app.memory_profiles import WINDOWS
from app.memory_store import StoreBinding,check_transition
from app.records import Records
from deploy.data.backup import create
from deploy.data.restore import restore
from deploy.memory.migrate import EmbeddingProfile,migrate
from deploy.memory.prepare import BINARY
from deploy.memory.reindex import reindex,abort
from test_mem0_adapter import FixtureEmbedding,actual_sdk
import test_memory_service as service_tests


class MatrixFixture(FixtureEmbedding):
    def embed(self,text,memory_action=None):return [super().embed(text,memory_action)]


@unittest.skipUnless(BINARY.exists(),'Pinned native Qdrant must be prepared')
class ReindexTests(unittest.TestCase):
    setUpClass=classmethod(service_tests.NativeMemoryBindingTests.setUpClass.__func__)
    tearDownClass=classmethod(service_tests.NativeMemoryBindingTests.tearDownClass.__func__)

    def setUp(self):
        self.app=self.root/uuid.uuid4().hex;self.path=self.app/'memory'
        with Mem0Adapter(self.path,sdk=actual_sdk(self.path),embedding_scope='FIXTURE') as memory:
            self.keep=memory.remember('owner','fact','keep','保留资料',source_id='original',request_id='keep')['memory_id']
            self.drop=memory.remember('owner','fact','drop','删除资料',source_id='obsolete',request_id='drop')['memory_id']
            self.guest=memory.remember('guest','fact','guest','访客资料',source_id='guest',request_id='guest')['memory_id']
            memory.sdk.entity_store.insert(vectors=[FixtureEmbedding().embed('资料')],ids=[str(uuid.uuid4())],payloads=[
                {'data':'资料','user_id':'owner','linked_memory_ids':[self.keep,self.drop]}])
        with Records(self.app/'records.sqlite'):pass
        migrate(self.path,self.service,EmbeddingProfile(FixtureEmbedding(),'f'*64,32,'FIXTURE'),records=self.app/'records.sqlite')
        self.original=json.loads((self.path/'store.json').read_text())
        self.profile=EmbeddingProfile(MatrixFixture(),'e'*64,32,'FIXTURE',WINDOWS)

    def marker(self):return json.loads((self.path/'reindex-transition.json').read_text())

    def test_reindex_preserves_ids_controls_and_applies_existing_deletion(self):
        with sqlite3.connect(self.path/'operations.sqlite') as db:
            db.execute("INSERT INTO deletion_barriers VALUES (?,?,1,'PENDING','FIXTURE')",(self.drop,'owner'))
        result=reindex(self.app,self.profile);self.assertEqual(result['status'],'PASS')
        binding=json.loads((self.path/'store.json').read_text())
        self.assertEqual(binding['schema_version'],2)
        with closing(self.service.client()) as client:
            self.assertFalse(client.collection_exists(self.original['collection']))
            points=client.retrieve(binding['collection'],[self.keep,self.drop,self.guest],with_vectors=True)
            self.assertEqual({str(x.id) for x in points},{self.keep,self.guest})
            self.assertTrue(all(len(x.vector)==1 for x in points))
            entity=client.scroll(binding['collection']+'_entities')[0][0]
            self.assertEqual(entity.payload['linked_memory_ids'],[self.keep])
        with sqlite3.connect(self.path/'history.sqlite') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM history WHERE memory_id=?',(self.drop,)).fetchone()[0],0)
        self.assertTrue(reindex(self.app,self.profile)['already_completed'])

    def test_interruptions_resume_through_publication_and_retired_cleanup(self):
        for phase in ('batch','verified','published'):
            with self.subTest(phase=phase):
                if phase!='batch':self.setUp()
                def crash(here):
                    if here==phase:raise RuntimeError('injected crash')
                with self.assertRaisesRegex(RuntimeError,'injected'):reindex(self.app,self.profile,checkpoint=crash)
                with self.assertRaisesRegex(RuntimeError,'reindex is incomplete'):check_transition(self.path)
                self.assertEqual(reindex(self.app,self.profile)['status'],'PASS')
                check_transition(self.path)

    def test_abort_restores_source_before_publish_and_rejects_after_publish(self):
        def crash(phase):
            if phase=='batch':raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):reindex(self.app,self.profile,checkpoint=crash)
        target=self.marker()['binding']['collection']
        self.assertEqual(abort(self.app)['status'],'ABORTED')
        self.assertEqual(json.loads((self.path/'store.json').read_text()),self.original)
        with closing(self.service.client()) as client:self.assertFalse(client.collection_exists(target))
        check_transition(self.path)
        def later(phase):
            if phase=='published':raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):reindex(self.app,self.profile,checkpoint=later)
        with self.assertRaisesRegex(RuntimeError,'cannot be aborted'):abort(self.app)
        reindex(self.app,self.profile)

    def test_late_source_change_refuses_resume_and_abort(self):
        def crash(phase):
            if phase=='batch':raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):reindex(self.app,self.profile,checkpoint=crash)
        with closing(self.service.client()) as client:
            client.set_payload(self.original['collection'],payload={'data':'unexpected edit'},points=[self.keep])
        with self.assertRaisesRegex(RuntimeError,'changed'):reindex(self.app,self.profile)
        with self.assertRaisesRegex(RuntimeError,'changed'):abort(self.app)

    def test_window_snapshot_restore_and_new_cutover_binding(self):
        reindex(self.app,self.profile)
        backup=self.root/uuid.uuid4().hex;target=self.root/uuid.uuid4().hex
        self.assertEqual(create(self.app,backup)['status'],'COMPLETE')
        self.assertEqual(restore(backup,self.app,target,self.service,self.profile)['status'],'PASS')
        saved=json.loads((target/'memory/store.json').read_text())
        owned=StoreBinding(target/'memory',self.service,embedding_sha256='e'*64,dimensions=32,profile=WINDOWS);owned.close()
        with closing(self.service.client()) as client:
            self.assertEqual(len(client.retrieve(saved['collection'],[self.keep],with_vectors=True)[0].vector),1)

    def test_old_scalar_backup_rebuilds_current_window_profile_without_reviving_forget(self):
        backup=self.root/uuid.uuid4().hex;target=self.root/uuid.uuid4().hex
        create(self.app,backup);reindex(self.app,self.profile)
        with sqlite3.connect(self.path/'operations.sqlite') as db:
            db.execute("INSERT INTO deletion_barriers VALUES (?,?,1,'PENDING','FIXTURE')",(self.drop,'owner'))
        def crash(phase):
            if phase=='snapshot_staged':raise RuntimeError('injected staged crash')
        with self.assertRaisesRegex(RuntimeError,'injected'):restore(backup,self.app,target,self.service,self.profile,checkpoint=crash)
        self.assertEqual(restore(backup,self.app,target,self.service,self.profile)['status'],'PASS')
        binding=json.loads((target/'memory/store.json').read_text())
        with closing(self.service.client()) as client:
            points=client.retrieve(binding['collection'],[self.keep,self.drop],with_vectors=True)
            self.assertEqual([str(p.id) for p in points],[self.keep])
            self.assertFalse(client.collection_exists(binding['collection']+'_source'))
            self.assertFalse(client.collection_exists(binding['collection']+'_entities_source'))
        owned=StoreBinding(target/'memory',self.service,embedding_sha256='e'*64,dimensions=32,profile=WINDOWS);owned.close()

    def test_cross_profile_restore_abort_removes_private_staging_only(self):
        from deploy.data.restore import abort as abort_restore
        from app.data_lifecycle import check_available
        backup=self.root/uuid.uuid4().hex;target=self.root/uuid.uuid4().hex
        create(self.app,backup);reindex(self.app,self.profile)
        current=json.loads((self.path/'store.json').read_text())
        def crash(phase):
            if phase=='snapshot_staged':raise RuntimeError('stop after staging')
        with self.assertRaises(RuntimeError):restore(backup,self.app,target,self.service,self.profile,checkpoint=crash)
        staged=json.loads((target/'data-transition.json').read_text())['staged_snapshots']
        self.assertEqual(abort_restore(target)['status'],'ABORTED')
        self.assertEqual(json.loads((self.path/'store.json').read_text()),current)
        check_available(self.app);check_transition(self.path)
        with closing(self.service.client()) as client:
            self.assertTrue(client.collection_exists(current['collection']))
            self.assertTrue(all(not client.collection_exists(name) for name in staged.values()))
