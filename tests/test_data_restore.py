"""Native snapshot restore with real SQLite/APScheduler and synthetic embeddings."""
from contextlib import closing
from datetime import datetime,timedelta,timezone
import json
import sqlite3
import unittest
import uuid

from qdrant_client import models
from app.data_lifecycle import check_available
from app.memory import Mem0Adapter
from app.memory_store import StoreBinding
from app.records import Records
from app.reminders import ReminderService,_deliver_due
from deploy.data.backup import create,sqlite_copy
from deploy.data.restore import abort,restore
from deploy.memory.migrate import EmbeddingProfile,migrate
from deploy.memory.prepare import BINARY
from test_mem0_adapter import FixtureEmbedding,actual_sdk
import test_memory_service as service_tests


@unittest.skipUnless(BINARY.exists(),'Pinned Qdrant must be prepared')
class ApplicationRestoreTests(unittest.TestCase):
    setUpClass=classmethod(service_tests.NativeMemoryBindingTests.setUpClass.__func__)
    tearDownClass=classmethod(service_tests.NativeMemoryBindingTests.tearDownClass.__func__)

    def setUp(self):
        self.source=self.root/uuid.uuid4().hex;self.backup=self.root/uuid.uuid4().hex;self.target=self.root/uuid.uuid4().hex
        self.profile=EmbeddingProfile(FixtureEmbedding(),'f'*64,32,'FIXTURE')
        memory=self.source/'memory'
        with Mem0Adapter(memory,sdk=actual_sdk(memory),embedding_scope='FIXTURE') as app:
            self.keep=app.remember('owner','preference','drink','绿茶',source_id='seed',request_id='save')['memory_id']
            self.forget=app.remember('owner','preference','obsolete','旧资料',source_id='old',request_id='old')['memory_id']
            self.child=app.remember('owner','summary','summary','派生旧资料',source_id='child',derived_from=(self.forget,),request_id='child')['memory_id']
            self.guest=app.remember('guest','preference','drink','咖啡',source_id='guest',request_id='guest')['memory_id']
            self.entity=str(uuid.uuid4())
            app.sdk.entity_store.insert(vectors=[FixtureEmbedding().embed('旧实体')],ids=[self.entity],payloads=[
                {'data':'旧实体','user_id':'owner','linked_memory_ids':[self.keep,self.forget]}])
        with Records(self.source/'records.sqlite') as records:
            records.begin('owner','session','old','旧资料');records.attach_memories('owner','old',[self.forget]);records.finish('owner','old','旧答案')
            records.create_note('owner','笔记','笔记正文',source_id='seed',request_id='note')
        with ReminderService(self.source/'reminders.sqlite') as reminders:
            due=datetime.now(timezone.utc)+timedelta(hours=1)
            self.future=reminders.create('owner','未来提醒',due,request_id='future')['reminder_id']
            self.cancel=reminders.create('owner','取消提醒',due,request_id='cancel')['reminder_id']
            self.delivered=reminders.create('owner','已输出提醒',due,request_id='delivered')['reminder_id']
        migrate(memory,self.service,self.profile,records=self.source/'records.sqlite')
        create(self.source,self.backup)
        self.binding=json.loads((memory/'store.json').read_text())

    def later_controls(self,*,correct=False):
        # These fixtures exercise real native persistence and current control
        # authority. Production writes use Mem0Adapter; values are labelled.
        with Records(self.source/'records.sqlite') as records:records.purge_memory('owner',self.forget)
        with ReminderService(self.source/'reminders.sqlite') as reminders:
            reminders.cancel('owner',self.cancel,request_id='later-cancel')
            _deliver_due(str(self.source/'reminders.sqlite'),self.delivered)
            notice=reminders.claim_next('owner')
            self.assertTrue(reminders.begin_output(notice['id'],attempt_id=notice['attempt_id'],channel='fixture'))
            self.assertTrue(reminders.acknowledge_delivery(notice['id'],attempt_id=notice['attempt_id'],evidence='fixture-drained'))
        if correct:
            with closing(self.service.client()) as client:
                point=client.retrieve(self.binding['collection'],[self.keep],with_vectors=True)[0]
                payload=dict(point.payload,data='桂花茶',version=2,operation_key='later-correct',source_id='later')
                client.upsert(self.binding['collection'],[models.PointStruct(id=point.id,vector=FixtureEmbedding().embed('桂花茶'),payload=payload)],wait=True)
            with sqlite3.connect(self.source/'memory/operations.sqlite') as db:
                receipt=json.dumps({'memory_id':self.keep,'version':2,'durable':True})
                db.execute('INSERT INTO operations VALUES (?,?,?,?,?,?,?,?,?)',('owner','later-correct','correct','fixture-fingerprint','COMPLETED',self.keep,receipt,datetime.now(timezone.utc).isoformat(),None))

    def run_restore(self,**kwargs):return restore(self.backup,self.source,self.target,self.service,self.profile,**kwargs)

    def test_current_deletions_versions_records_and_reminder_receipts_win(self):
        self.later_controls(correct=True)
        result=self.run_restore()
        self.assertEqual(result['memories'],2);self.assertEqual(result['deleted_ids'],2)
        target=json.loads((self.target/'memory/store.json').read_text())
        with closing(self.service.client()) as client:
            self.assertEqual(client.retrieve(target['collection'],[self.forget,self.child]),[])
            point=client.retrieve(target['collection'],[self.keep])[0]
            self.assertEqual((point.payload['data'],point.payload['version']),('桂花茶',2))
            self.assertEqual(client.retrieve(target['collection'],[self.guest])[0].payload['user_id'],'guest')
            self.assertEqual(client.retrieve(target['collection']+'_entities',[self.entity]),[])
        binding=StoreBinding(self.target/'memory',self.service,embedding_sha256='f'*64,dimensions=32);binding.close()
        with Records(self.target/'records.sqlite') as records:
            self.assertEqual(records.recent('owner'),[]);self.assertEqual(len(records.list_notes('owner')),1)
        with ReminderService(self.target/'reminders.sqlite') as reminders:
            self.assertIsNone(reminders.claim_next('owner'))
            self.assertEqual([job.id for job in reminders.scheduler.get_jobs()],['reminder:'+self.future])
            self.assertEqual(reminders.scheduler.get_jobs()[0].args[0],str(self.target/'reminders.sqlite'))
        with self.assertRaisesRegex(RuntimeError,'retired'):Mem0Adapter(self.source/'memory')
        with self.assertRaisesRegex(RuntimeError,'retired'):Records(self.source/'records.sqlite')
        with self.assertRaisesRegex(RuntimeError,'retired'):ReminderService(self.source/'reminders.sqlite')
        self.assertTrue(self.run_restore()['already_completed'])

    def test_missing_vector_index_is_rebuilt_without_resurrecting_deleted_facts(self):
        self.later_controls()
        with closing(self.service.client()) as client:client.delete_collection(self.binding['collection'])
        self.assertEqual(self.run_restore()['memories'],2)

    def test_unbacked_newer_version_causes_refusal(self):
        self.later_controls(correct=True)
        with closing(self.service.client()) as client:client.delete_collection(self.binding['collection'])
        with self.assertRaisesRegex(RuntimeError,'version is unavailable'):self.run_restore()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):check_available(self.target)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):Records(self.source/'records.sqlite')
        self.assertEqual(abort(self.target)['status'],'ABORTED')
        check_available(self.source)

    def test_corrupt_current_vector_can_use_same_confirmed_version_from_snapshot(self):
        with closing(self.service.client()) as client:
            point=client.retrieve(self.binding['collection'],[self.keep],with_vectors=True)[0]
            client.upsert(self.binding['collection'],[models.PointStruct(id=point.id,vector=FixtureEmbedding().embed('损坏向量'),payload=point.payload)],wait=True)
        self.run_restore()
        binding=json.loads((self.target/'memory/store.json').read_text())
        with closing(self.service.client()) as client:
            point=client.retrieve(binding['collection'],[self.keep],with_vectors=True)[0]
            from deploy.memory.migrate import matching
            self.assertTrue(matching(point.vector,FixtureEmbedding().embed('绿茶'),32))

    def test_missing_dense_vector_can_be_recovered_from_snapshot(self):
        with closing(self.service.client()) as client:
            client.delete_vectors(self.binding['collection'],vectors=[''],points=[self.keep],wait=True)
        self.assertEqual(self.run_restore()['status'],'PASS')

    def test_notification_state_cannot_regress_to_unplayed(self):
        with ReminderService(self.source/'reminders.sqlite') as reminders:
            _deliver_due(str(self.source/'reminders.sqlite'),self.delivered)
            notice=reminders.claim_next('owner');reminders.begin_output(notice['id'],attempt_id=notice['attempt_id'],channel='fixture')
        self.backup=self.root/uuid.uuid4().hex;create(self.source,self.backup)
        # Labelled older/wrong authority: an output already started in backup.
        with sqlite3.connect(self.source/'reminders.sqlite') as db:db.execute("UPDATE assistant_notifications SET status='pending'")
        with self.assertRaisesRegex(RuntimeError,'notification delivery state regressed'):self.run_restore()
        self.assertFalse((self.source/'data-transition.json').exists())

    def test_interrupted_cutover_resumes_and_never_reopens_both_sides(self):
        for phase in ('target_marked','source_fenced','snapshot','merged','verified','source_retired'):
            with self.subTest(phase=phase):
                if phase!='target_marked':self.setUp()
                self.later_controls()
                def fail(value):
                    if value==phase:raise OSError('LABELLED restore interruption')
                with self.assertRaisesRegex(OSError,'LABELLED'):self.run_restore(checkpoint=fail)
                with self.assertRaisesRegex(RuntimeError,'incomplete'):check_available(self.target)
                if phase!='target_marked':
                    with self.assertRaisesRegex(RuntimeError,'incomplete|retired'):Records(self.source/'records.sqlite')
                self.assertEqual(self.run_restore()['status'],'PASS')

    def test_source_control_change_after_interruption_is_rejected(self):
        def fail(phase):
            if phase=='source_fenced':raise OSError('LABELLED')
        with self.assertRaises(OSError):self.run_restore(checkpoint=fail)
        with sqlite3.connect(self.source/'records.sqlite') as db:db.execute("UPDATE notes SET body='LABELLED external change'")
        with self.assertRaisesRegex(RuntimeError,'controls changed'):self.run_restore()

    def test_other_lineage_and_busy_source_are_rejected_before_fencing(self):
        with Records(self.source/'records.sqlite'):
            with self.assertRaisesRegex(RuntimeError,'in use'):self.run_restore()
        self.assertFalse((self.source/'data-transition.json').exists())
        manifest=json.loads((self.backup/'manifest.json').read_text());manifest['lineage_id']=uuid.uuid4().hex
        (self.backup/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError,'different lineages'):self.run_restore()

    def test_abort_resumes_after_source_retirement_but_rejects_active_target(self):
        def fail(phase):
            if phase=='source_retired':raise OSError('LABELLED')
        with self.assertRaises(OSError):self.run_restore(checkpoint=fail)
        def fail_abort(phase):
            if phase=='target_aborted':raise OSError('LABELLED')
        with self.assertRaises(OSError):abort(self.target,checkpoint=fail_abort)
        self.assertEqual(abort(self.target)['status'],'ABORTED');check_available(self.source)
        self.target=self.root/uuid.uuid4().hex;self.run_restore()
        with self.assertRaisesRegex(RuntimeError,'unpublished'):abort(self.target)

    def test_second_cutover_preserves_lineage_and_resume_before_source_fence(self):
        self.run_restore();first_target=self.target
        self.source=first_target;self.target=self.root/uuid.uuid4().hex
        def fail(phase):
            if phase=='target_marked':raise OSError('LABELLED')
        with self.assertRaises(OSError):self.run_restore(checkpoint=fail)
        self.run_restore();check_available(self.target)
        with self.assertRaisesRegex(RuntimeError,'retired'):check_available(first_target)
