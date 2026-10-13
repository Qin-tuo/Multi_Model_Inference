"""Calendar/role provenance with real Mem0/Qdrant, explicit fixture vectors."""
from datetime import datetime,timezone
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
import asyncio
import threading
from types import SimpleNamespace

from app.memory import Mem0Adapter,MemoryAdapterError
from app.memory_sources import constraints,dialogue_lookup,provenance
from test_mem0_adapter import actual_sdk


class SourceQueryTests(unittest.TestCase):
    def test_companion_pronoun_does_not_replace_the_speaking_subject(self):
        for link in ('跟','与','同','和','向','对','给'):
            with self.subTest(link=link):
                self.assertEqual(dialogue_lookup(f'4月27日，我{link}你分享了什么书？'),{'source_month_day':'04-27','source_speaker':'user'})
                self.assertEqual(dialogue_lookup(f'4月27日，你{link}我分享了什么书？'),{'source_month_day':'04-27','source_speaker':'assistant'})
        self.assertEqual(dialogue_lookup('4月27日，我跟你分享了什么，你又向我推荐了什么？'),{'source_month_day':'04-27'})

    def test_explicit_calendar_and_speaker_without_invented_year(self):
        self.assertEqual(dialogue_lookup('2023年4月30日，我分享了一句诗，内容是什么？'),{'source_date':'2023-04-30','source_speaker':'user'})
        self.assertEqual(dialogue_lookup('4月30日，我分享了一句诗，内容是什么？'),{'source_month_day':'04-30','source_speaker':'user'})
        self.assertEqual(dialogue_lookup('二零二三年四月三十日，你向我建议了哪些方法？'),{'source_date':'2023-04-30','source_speaker':'assistant'})
        self.assertEqual(dialogue_lookup('2023-04-30我们聊到了什么？'),{'source_date':'2023-04-30'})

    def test_event_time_range_and_new_statements_are_not_source_dates(self):
        for text in ('2023年5月6日我去市场买了什么？','记住4月27日我们聊了这件事。',
                     '4月27日到4月30日我们聊了什么？','4月30日之前我提到过哪些书？','昨天我们聊了什么？',
                     '去年4月27日我分享了什么？'):
            with self.subTest(text=text):self.assertEqual(dialogue_lookup(text),{})
        self.assertEqual(dialogue_lookup('5月9日，你向我建议了哪些因素至关重要？'),{'source_month_day':'05-09','source_speaker':'assistant'})

    def test_invalid_metadata_and_conflicting_calendar_are_rejected(self):
        for value in ('2023-02-29','2023-4-2','2023-04-02T00:00:00',True):
            with self.subTest(value=value),self.assertRaises((ValueError,TypeError)):provenance(value,'user')
        with self.assertRaises(ValueError):provenance('2023-04-30','system')
        with self.assertRaises(ValueError):constraints('2023-04-30','05-01')

    def test_invalid_spoken_dates_do_not_abort_a_conversation(self):
        for text in ('2023-02-29我分享了什么？','2023年13月4日我说过什么？','二零二三年二月三十日我推荐了什么？'):
            with self.subTest(text=text):self.assertEqual(dialogue_lookup(text),{})


class SourceStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.memory=Mem0Adapter(self.path,sdk=actual_sdk(self.path),embedding_scope='FIXTURE')
    def tearDown(self):self.memory.close();self.temp.cleanup()
    def restart(self):
        self.memory.close();self.memory=Mem0Adapter(self.path,sdk=actual_sdk(self.path),embedding_scope='FIXTURE')
    def add(self,key,day,speaker='user',owner='owner',namespace='personal'):
        return self.memory.remember(owner,'event',key,'我们聊到了同一本书。',source_id=key,request_id=key,
            namespace=namespace,source_date=day,source_speaker=speaker)['memory_id']

    def test_native_filters_apply_date_speaker_owner_namespace_before_topk(self):
        wanted=self.add('wanted','2023-04-30')
        self.add('wrong-date','2023-05-01');self.add('wrong-speaker','2023-04-30','assistant')
        self.add('wrong-owner','2023-04-30',owner='guest');self.add('wrong-namespace','2023-04-30',namespace='other')
        self.restart()
        found=self.memory.search('owner','我们聊到了同一本书。',source_date='2023-04-30',source_speaker='user',top_k=1)
        self.assertEqual([item['memory_id'] for item in found['items']],[wanted])
        self.assertEqual(found['items'][0]['sources'][0]['date'],'2023-04-30')
        self.assertEqual(self.memory.search('owner',key='wanted',source_date='2023-05-01')['items'],[])
        self.memory.forget('owner',wanted,request_id='forget')
        self.assertEqual(self.memory.search('owner','我们聊到了同一本书。',source_date='2023-04-30',source_speaker='user')['items'],[])

    def test_month_day_retains_years_in_returned_provenance(self):
        self.add('year-2023','2023-04-30');self.add('year-2024','2024-04-30')
        found=self.memory.search('owner','我们聊到了同一本书。',source_month_day='04-30')['items']
        self.assertEqual({item['sources'][0]['date'] for item in found},{'2023-04-30','2024-04-30'})

    def test_provenance_is_in_idempotency_and_correction_replaces_source(self):
        mid=self.add('same','2023-04-30','assistant')
        with self.assertRaises(MemoryAdapterError) as error:self.add('same','2023-05-01','assistant')
        self.assertEqual(error.exception.code,'CONFLICT')
        result=self.memory.correct('owner',mid,'更正后的书名',expected_version=1,source_id='correction',request_id='correct',
            source_date='2023-05-02',source_speaker='user')
        self.assertTrue(result['durable'])
        self.assertEqual(self.memory.search('owner',key='same',source_date='2023-04-30')['items'],[])
        self.assertEqual(self.memory.search('owner',key='same',source_date='2023-05-02')['items'][0]['sources'][0]['speaker'],'user')
        self.memory.correct('owner',mid,'未注明日期的新来源',expected_version=2,source_id='unknown',request_id='unknown')
        source=self.memory.search('owner',key='same')['items'][0]['sources'][0]
        self.assertNotIn('date',source);self.assertNotIn('speaker',source)

    def test_legacy_calls_remain_replayable(self):
        args=dict(source_id='legacy',request_id='legacy')
        first=self.memory.remember('owner','fact','legacy','旧调用',**args)
        self.restart();second=self.memory.remember('owner','fact','legacy','旧调用',**args)
        self.assertEqual(first['memory_id'],second['memory_id']);self.assertTrue(second['replayed'])


class SourceTranscriptTests(unittest.TestCase):
    def test_local_day_boundaries_owner_and_completed_status(self):
        from app.records import Records
        from sqlalchemy import text
        with tempfile.TemporaryDirectory() as tmp,patch('app.records.now',return_value=datetime(2026,10,14,tzinfo=timezone.utc)):
            with Records(Path(tmp)/'records.sqlite') as records:
                for turn,owner,stamp in [('before','owner','2026-10-12T15:59:59+00:00'),
                        ('start','owner','2026-10-12T16:00:00+00:00'),('end','owner','2026-10-13T15:59:59+00:00'),
                        ('after','owner','2026-10-13T16:00:00+00:00'),('guest','guest','2026-10-13T00:00:00+00:00')]:
                    records.begin(owner,'session',turn,'问题');records.finish(owner,turn,'回答')
                    with records.engine.begin() as db:db.execute(text('UPDATE turns SET created_at=:stamp WHERE id=:id'),{'stamp':stamp,'id':turn})
                selected=records.recent('owner',source_date='2026-10-13')
                self.assertEqual([row['id'] for row in selected],['start','end'])
                self.assertEqual([row['id'] for row in records.recent('owner',source_month_day='10-13')],['start','end'])
                self.assertEqual(records.recent('owner',source_date='2023-10-13'),[])


class SourceParserAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_parser_does_not_block_the_pipeline_event_loop(self):
        from app.session import AssistantSession
        started=threading.Event();release=threading.Event()
        def parser(_text):
            started.set()
            if not release.wait(2):raise AssertionError('Source parser blocked the event loop')
            return {}
        class ReachedSearch(RuntimeError):pass
        def search(*args,**kwargs):raise ReachedSearch()
        session=SimpleNamespace(memory=SimpleNamespace(search=search),user_id='owner',memory_top_k=5,memory_context_tokens=512)
        turn=SimpleNamespace(requested_lookup=lambda:None,required_operation=lambda:None,cancelled=False)
        with patch('app.memory_sources.dialogue_lookup',parser):
            task=asyncio.create_task(AssistantSession._messages(session,'4月27日我分享了什么？',turn))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait,1))
                release.set()
                with self.assertRaises(ReachedSearch):await task
            finally:
                release.set()
                if not task.done():task.cancel()
