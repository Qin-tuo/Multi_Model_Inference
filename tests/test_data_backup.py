"""Real Qdrant snapshot and SQLite backup roundtrip; synthetic SDK embeddings."""
from contextlib import closing
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sqlite3
import unittest
import uuid
from unittest.mock import patch

from qdrant_client import models
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService,_deliver_due
from deploy.data.backup import collection_state,create,sqlite_copy,sqlite_state,upload_snapshot_to_new_collection,validate
from deploy.memory.migrate import EmbeddingProfile,migrate
from deploy.memory.prepare import BINARY
from deploy.memory.runtime import private_key
from test_mem0_adapter import FixtureEmbedding,actual_sdk
import test_memory_service as service_tests


@unittest.skipUnless(BINARY.exists(),'Pinned native Qdrant must be prepared')
class ApplicationBackupTests(unittest.TestCase):
    setUpClass=classmethod(service_tests.NativeMemoryBindingTests.setUpClass.__func__)
    tearDownClass=classmethod(service_tests.NativeMemoryBindingTests.tearDownClass.__func__)

    def setUp(self):
        self.app=self.root/uuid.uuid4().hex;self.memory=self.app/'memory'
        self.backup=self.root/uuid.uuid4().hex
        with Mem0Adapter(self.memory,sdk=actual_sdk(self.memory),embedding_scope='FIXTURE') as memory:
            self.saved=memory.remember('owner','preference','drink','测试饮品',source_id='source',request_id='save')['memory_id']
            self.guest=memory.remember('guest','preference','drink','访客饮品',source_id='guest')['memory_id']
            memory.sdk.entity_store.insert(vectors=[FixtureEmbedding().embed('饮品')],ids=[str(uuid.uuid4())],payloads=[
                {'data':'饮品','user_id':'owner','linked_memory_ids':[self.saved]}])
        migrate(self.memory,self.service,EmbeddingProfile(FixtureEmbedding(),'f'*64,32,'FIXTURE'))
        with Records(self.app/'records.sqlite') as records:
            records.begin('owner','session','turn','测试对话');records.finish('owner','turn','测试回答')
            records.create_note('owner','测试标题','原始内容',source_id='turn',request_id='note')
        with ReminderService(self.app/'reminders.sqlite') as reminders:
            self.reminder=reminders.create('owner','测试提醒',datetime.now(timezone.utc)+timedelta(hours=1),request_id='reminder')['reminder_id']
            self.cancelled=reminders.create('owner','取消提醒',datetime.now(timezone.utc)+timedelta(hours=1),request_id='cancelled')['reminder_id']
            reminders.cancel('owner',self.cancelled,request_id='cancel')

    def test_native_snapshots_and_all_sqlite_roundtrip(self):
        # A committed WAL remains present while this labelled SQLite handle is
        # open; copying only the base .sqlite file would lose this update.
        with closing(sqlite3.connect(self.app/'records.sqlite')) as live:
            live.execute('PRAGMA journal_mode=WAL')
            live.execute("UPDATE notes SET body='WAL中的最新内容'");live.commit()
            report=create(self.app,self.backup)
            self.assertEqual(report['status'],'COMPLETE')
        self.assertEqual(set(report['databases']),{'memory/operations.sqlite','memory/history.sqlite','records.sqlite','reminders.sqlite'})
        for name,expected in report['databases'].items():
            restored=self.root/(uuid.uuid4().hex+'.sqlite');sqlite_copy(self.backup/name,restored)
            self.assertEqual(sqlite_state(restored),expected)
        with closing(sqlite3.connect(self.backup/'records.sqlite')) as db:
            self.assertEqual(db.execute('SELECT body FROM notes').fetchone()[0],'WAL中的最新内容')
        with closing(self.service.client(timeout=60)) as client:
            for role,expected in report['collections'].items():
                restored='restore_probe_'+uuid.uuid4().hex
                self.assertFalse(client.collection_exists(restored))
                try:
                    upload_snapshot_to_new_collection(client,restored,self.backup/expected['file'],report['files'][expected['file']]['sha256'])
                    self.assertEqual(collection_state(client,restored),{k:expected[k] for k in ('sha256','points')})
                    with self.assertRaisesRegex(RuntimeError,'refusing to overwrite'):
                        upload_snapshot_to_new_collection(client,restored,self.backup/expected['file'],report['files'][expected['file']]['sha256'])
                finally:
                    if client.collection_exists(restored):client.delete_collection(restored)
            for value in report['collections'].values():self.assertEqual(client.list_snapshots(value['collection']),[])
        self.assertEqual(validate(self.backup)['backup_id'],report['backup_id'])
        self.assertEqual(report['server_snapshots'],[])
        for name in report['files']:self.assertEqual((self.backup/name).stat().st_mode&0o777,0o600)
        self.assertFalse(any('api-key' in p.name or p.name=='deployment.json' for p in self.backup.rglob('*')))
        credential=private_key(self.service.key_path).encode()
        self.assertFalse(any(credential in (self.backup/name).read_bytes() for name in report['files']),'Credential content must not enter a backup')

    def test_busy_owners_rejected_and_completed_backup_immutable(self):
        with Records(self.app/'records.sqlite'):
            with self.assertRaisesRegex(RuntimeError,'in use'):create(self.app,self.backup)
        self.assertFalse((self.backup/'manifest.json').exists())
        with ReminderService(self.app/'reminders.sqlite'):
            with self.assertRaisesRegex(RuntimeError,'in use'):create(self.app,self.backup)
        create(self.app,self.backup)
        with self.assertRaisesRegex(RuntimeError,'immutable'):create(self.app,self.backup)

    def test_partial_backup_resumes_but_is_not_usable_before_completion(self):
        def fail(phase):
            if phase=='snapshot':raise OSError('LABELLED transfer interruption')
        with self.assertRaisesRegex(OSError,'LABELLED'):create(self.app,self.backup,checkpoint=fail)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):validate(self.backup)
        first=json.loads((self.backup/'manifest.json').read_text())['backup_id']
        # A new consistent cut is taken on retry, including intervening writes.
        with Records(self.app/'records.sqlite') as records:
            records.create_note('owner','重试前新增','新增内容',source_id='new',request_id='new')
        result=create(self.app,self.backup)
        self.assertEqual(result['backup_id'],first)
        with closing(sqlite3.connect(self.backup/'records.sqlite')) as db:self.assertEqual(db.execute('SELECT count(*) FROM notes').fetchone()[0],2)

    def test_checksums_paths_and_unknown_directory_are_rejected(self):
        create(self.app,self.backup)
        target=self.backup/'memory/history.sqlite'
        with target.open('ab') as file:file.write(b'LABELLED corruption')
        with self.assertRaisesRegex(RuntimeError,'checksum'):validate(self.backup)
        unrelated=self.root/uuid.uuid4().hex;unrelated.mkdir();(unrelated/'user.txt').write_text('keep')
        with self.assertRaisesRegex(RuntimeError,'not empty'):create(self.app,unrelated)
        self.assertEqual((unrelated/'user.txt').read_text(),'keep')
        with self.assertRaisesRegex(RuntimeError,'separate'):create(self.app,self.app/'backup')

    def test_disk_failure_and_late_corruption_never_publish_complete(self):
        with patch('deploy.data.backup.sqlite_copy',side_effect=OSError('LABELLED disk full')):
            with self.assertRaisesRegex(OSError,'disk full'):create(self.app,self.backup)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):validate(self.backup)
        def corrupt(phase):
            if phase=='verified':
                with (self.backup/'memory.snapshot').open('ab') as file:file.write(b'LABELLED late corruption')
        with self.assertRaisesRegex(RuntimeError,'changed before publication'):create(self.app,self.backup,checkpoint=corrupt)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):validate(self.backup)

    def test_source_mutation_is_detected(self):
        def mutate(phase):
            if phase=='snapshot':
                # Labelled bypass of the application writer locks to prove the
                # end-of-backup consistency check, not a supported write path.
                with sqlite3.connect(self.app/'records.sqlite') as db:db.execute("UPDATE notes SET body=body||'changed'")
        with self.assertRaisesRegex(RuntimeError,'changed during backup'):create(self.app,self.backup,checkpoint=mutate)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):validate(self.backup)
