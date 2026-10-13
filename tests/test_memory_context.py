"""Keep source provenance while packing nonduplicate evidence into a token cap."""
from io import BytesIO
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
