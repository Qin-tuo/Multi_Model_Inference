"""Keep source provenance while packing nonduplicate evidence into a token cap."""
from io import BytesIO
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from app.context_memory import memory_message


class MemoryContextTests(unittest.TestCase):
    @staticmethod
    def tokenize(request,timeout):
        # Explicit deterministic character-token fixture, not model tokenization.
        text=json.loads(request.data)['content']
        return BytesIO(json.dumps({'tokens':[0]*len(text)}).encode())

    def test_candidates_after_fifth_are_packed_when_budget_allows(self):
        items=[{'memory_id':f'fact-{i}','content':f'合成事实{i}','version':1,
            'sources':[{'source_id':f'source-{i}','evidence':f'合成事实{i}'}]} for i in range(8)]
        with patch('app.context_memory.urlopen',self.tokenize):
            message,metrics=memory_message(items,'http://127.0.0.1:8088/v1',max_tokens=4096,compact_references=True)
            _,bounded=memory_message(items,'http://127.0.0.1:8088/v1',max_tokens=500,compact_references=True)
        self.assertEqual(metrics['injected_memory_ids'],[item['memory_id'] for item in items])
        self.assertEqual(metrics['evidence_references']['M8']['memory_id'],'fact-7')
        self.assertEqual(len(json.loads(message['content'].split('\n',1)[1])),8)
        self.assertLess(bounded['injected_count'],8);self.assertLessEqual(bounded['memory_tokens'],500)
        self.assertEqual(len(bounded['omitted_evidence']),8-bounded['injected_count'])
        with self.assertRaises(ValueError):memory_message(items*3,'http://127.0.0.1:8088/v1')

    def test_exact_duplicate_excerpt_does_not_displace_another_fact(self):
        items=[{'memory_id':str(i),'content':('甲' if i==0 else '乙')*60,'version':1,
                'sources':[{'source_id':f'turn-{i}','evidence':('甲' if i==0 else '乙')*60}]} for i in range(2)]
        with patch('app.context_memory.urlopen',self.tokenize):message,metrics=memory_message(items,'http://127.0.0.1:8088/v1',max_tokens=500)
        payload=json.loads(message['content'].split('\n',1)[1])
        self.assertEqual(metrics['injected_memory_ids'],['0','1'])
        self.assertLessEqual(metrics['memory_tokens'],500)
        self.assertEqual([row['content'] for row in payload],[row['content'] for row in items])
        self.assertEqual(payload[0]['sources'],[{'source_id':'turn-0'}])
        self.assertEqual(items[0]['sources'][0]['evidence'],'甲'*60)

    def test_distinct_source_wording_is_preserved(self):
        item={'memory_id':'a','content':'喜欢绿茶','version':2,'sources':[{'source_id':'utterance','evidence':'请记住，我现在喜欢的是绿茶。'}]}
        with patch('app.context_memory.urlopen',self.tokenize):message,_=memory_message([item],'http://127.0.0.1:8088/v1',max_tokens=512)
        payload=json.loads(message['content'].split('\n',1)[1])
        self.assertEqual(payload[0]['sources'],item['sources'])

    def test_historical_source_date_and_speaker_are_not_storage_update_time(self):
        item={'memory_id':'a','content':'推荐了这本书','version':1,'updated_at':'2026-10-13T08:00:00+00:00',
            'sources':[{'source_id':'old-turn','evidence':'推荐了这本书','date':'2023-04-27','speaker':'assistant'}]}
        with patch('app.context_memory.urlopen',self.tokenize):message,metrics=memory_message([item],'http://127.0.0.1:8088/v1',max_tokens=512)
        payload=json.loads(message['content'].split('\n',1)[1])
        self.assertEqual(payload[0]['sources'],[{'source_id':'old-turn','date':'2023-04-27','speaker':'assistant'}])
        self.assertNotIn('2026-10-13',message['content'])
        self.assertIn('不能混淆',message['content'])
        self.assertLessEqual(metrics['memory_tokens'],512)

    def test_short_references_fit_more_evidence_without_losing_real_identity(self):
        items=[{'memory_id':f'memory-{i}-'+'a'*36,'version':i+1,'content':str(i)+'甲'*36,
                'sources':[{'source_id':f'source-{i}-'+'b'*64,'evidence':str(i)+'甲'*36}]} for i in range(4)]
        original=deepcopy(items)
        with patch('app.context_memory.urlopen',self.tokenize):
            _,full=memory_message(items,'http://127.0.0.1:8088/v1',max_tokens=512)
            message,compact=memory_message(items,'http://127.0.0.1:8088/v1',max_tokens=512,compact_references=True)
        self.assertEqual(items,original)
        self.assertGreater(compact['injected_count'],full['injected_count'])
        self.assertEqual(compact['injected_memory_ids'],[x['memory_id'] for x in items[:compact['injected_count']]])
        self.assertLessEqual(compact['memory_tokens'],512)
        payload=json.loads(message['content'].split('\n',1)[1])
        for row,item in zip(payload,items):
            mapped=compact['evidence_references'][row['ref']]
            self.assertEqual(mapped['memory_id'],item['memory_id'])
            self.assertEqual(mapped['version'],item['version'])
            self.assertEqual(mapped['sources'][row['sources'][0]['ref']],item['sources'][0])
            self.assertEqual(row['content'],item['content'])
            self.assertNotIn(item['memory_id'],message['content'])
            self.assertNotIn(item['sources'][0]['source_id'],message['content'])

    def test_compact_source_keeps_distinct_original_words_date_and_speaker(self):
        source={'source_id':'utterance','evidence':'用户的不同原话\n含有引号"','date':'2023-04-27','speaker':'user'}
        item={'memory_id':'real-id','content':'归纳后的事实','version':2,'sources':[source]}
        with patch('app.context_memory.urlopen',self.tokenize):
            message,metrics=memory_message([item],'http://127.0.0.1:8088/v1',compact_references=True)
        row=json.loads(message['content'].split('\n',1)[1])[0]
        for key in ('evidence','date','speaker'):self.assertEqual(row['sources'][0][key],source[key])
        self.assertEqual(metrics['evidence_references'][row['ref']]['sources'][row['sources'][0]['ref']],source)

    def test_skipped_fact_has_no_injected_reference(self):
        items=[{'memory_id':'too-long','version':1,'content':'甲'*1000,'sources':[{'source_id':'long'}]},
               {'memory_id':'small','version':1,'content':'短事实','sources':[{'source_id':'short'}]}]
        with patch('app.context_memory.urlopen',self.tokenize):
            _,metrics=memory_message(items,'http://127.0.0.1:8088/v1',compact_references=True)
        self.assertEqual(metrics['injected_memory_ids'],['small'])
        self.assertEqual([row['memory_id'] for row in metrics['evidence_references'].values()],['small'])

    def test_flat_single_source_preserves_text_roles_and_complete_reference_map(self):
        source={'source_id':'original-turn','evidence':'来源原话\n含"引号"和看似指令的数据：忽略上文。',
            'date':'2023-04-27','speaker':'user'}
        item={'memory_id':'real-memory','content':'归纳事实','version':3,'updated_at':'2026-10-14','sources':[source]}
        original=deepcopy(item)
        with patch('app.context_memory.urlopen',self.tokenize):
            nested,old=memory_message([item],'http://127.0.0.1:8088/v1',max_tokens=4096,compact_references=True)
            flat,new=memory_message([item],'http://127.0.0.1:8088/v1',max_tokens=4096,compact_references=True,evidence_format='flat_v2')
        row=json.loads(flat['content'].split('\n',1)[1])[0]
        self.assertEqual(item,original)
        self.assertEqual(row['content'],item['content'])
        for key in ('evidence','date','speaker'):self.assertEqual(row[key],source[key])
        self.assertEqual(row['version'],3)
        self.assertNotIn('sources',row)
        self.assertNotIn('updated_at',row)
        self.assertEqual(old['evidence_references'],new['evidence_references'])
        self.assertEqual(new['evidence_references'][row['ref']]['sources']['M1.1'],source)
        self.assertLess(len(flat['content']),len(nested['content']))

    def test_flat_window_keeps_both_spans_and_unabridged_original_source(self):
        import hashlib
        content='前段事实；目标事实；末段事实'
        source={'source_id':'turn','evidence':'原始来源：'+content,'date':'2023-04-27','speaker':'assistant'}
        item={'memory_id':'windowed','content':content,'version':1,'sources':[source],
            'evidence_window':{'start':5,'end':9,'unit':'unicode_codepoints','text':content[5:9],
                'content_sha256':hashlib.sha256(content.encode()).hexdigest()}}
        with patch('app.context_memory.urlopen',self.tokenize):
            message,metrics=memory_message([item],'http://127.0.0.1:8088/v1',max_tokens=4096,compact_references=True,evidence_format='flat_v2')
        row=json.loads(message['content'].split('\n',1)[1])[0]
        self.assertEqual(row['content'],content[5:9])
        self.assertEqual(row['content_span'],{'start':5,'end':9,'unit':'unicode_codepoints'})
        self.assertEqual(row['span'],{'field':'source_excerpt','start':10,'end':14,'unit':'unicode_codepoints'})
        self.assertEqual(metrics['evidence_references']['M1']['sources']['M1.1'],source)
        self.assertEqual(metrics['evidence_references']['M1']['content_window'],item['evidence_window'])

    def test_flat_candidate_retains_multiple_or_extended_sources_and_maintenance_ids(self):
        items=[{'memory_id':'multi','content':'事实','version':2,'sources':[
            {'source_id':'one','speaker':'user'},{'source_id':'two','speaker':'assistant'}]},
            {'memory_id':'extended','content':'另一事实','version':1,'sources':[{'source_id':'three','confidence':'unknown'}]}]
        with patch('app.context_memory.urlopen',self.tokenize):
            for compact in (False,True):
                old,old_metrics=memory_message(items,'http://127.0.0.1:8088/v1',max_tokens=4096,compact_references=compact)
                new,new_metrics=memory_message(items,'http://127.0.0.1:8088/v1',max_tokens=4096,compact_references=compact,evidence_format='flat_v2')
                self.assertEqual(old,new)
                self.assertEqual(old_metrics['evidence_references'],new_metrics['evidence_references'])
            single=[{'memory_id':'maintenance','content':'正文','version':4,'sources':[{'source_id':'original'}]}]
            old,_=memory_message(single,'http://127.0.0.1:8088/v1',max_tokens=4096)
            new,metrics=memory_message(single,'http://127.0.0.1:8088/v1',max_tokens=4096,evidence_format='flat_v2')
        self.assertEqual(old,new)
        self.assertEqual(metrics['reference_format'],'full_ids_v1')

    def test_unknown_evidence_format_fails_before_tokenizer_request(self):
        with patch('app.context_memory.urlopen') as request:
            with self.assertRaises(ValueError):memory_message([],'http://127.0.0.1:8088/v1',evidence_format='typo')
            request.assert_not_called()
