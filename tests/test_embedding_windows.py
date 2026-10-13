"""Window coverage and exact source identity; tokenizer/model here are fixtures."""
import hashlib
from types import SimpleNamespace
import unittest
import numpy as np

from app.adapters.embedding_windows import BGEWindowAdapter
from app.context_memory import presentation


class CharacterTokenizer:
    is_fast=True
    def __call__(self,text,**kwargs):return {'offset_mapping':[(i,i+1) for i in range(len(text))]}


class RecordingModel:
    tokenizer=CharacterTokenizer();max_seq_length=512
    def __init__(self):self.seen=[]
    def encode(self,texts,**kwargs):
        self.seen.extend(texts)
        return np.array([[1.,0.] if '目标' in text else [0.,1.] for text in texts])


class EmbeddingWindowTests(unittest.TestCase):
    def setUp(self):
        self.model=RecordingModel();self.encoder=BGEWindowAdapter(SimpleNamespace(model=self.model,config=None))

    def test_unicode_source_fully_covered_and_windows_bounded(self):
        text=('中文𠀀🙂\nＡ '*100)+'目标尾部'
        spans=self.encoder.spans(text);covered=set()
        for start,end in spans:
            self.assertLessEqual(end-start,256);covered.update(range(start,end))
        self.assertEqual(covered,set(range(len(text))))
        self.assertEqual(spans[0][0],0);self.assertEqual(spans[-1][1],len(text))
        matrices=self.encoder.embed_batch([text,'短内容'])
        self.assertEqual([len(v) for v in matrices],[len(spans),1])
        self.assertTrue(all(len(chunk)<=256 for chunk in self.model.seen))

    def test_matching_vector_selects_original_tail_with_verifiable_offsets(self):
        text='背景'*350+'目标尾部'
        window=self.encoder.best_window(text,self.encoder.embed(text,'add'),'目标是什么')
        self.assertIn('目标尾部',window['text'])
        self.assertEqual(window['text'],text[window['start']:window['end']])
        self.assertEqual(window['content_sha256'],hashlib.sha256(text.encode()).hexdigest())
        with self.assertRaises(ValueError):self.encoder.best_window(text,[[1.,0.]],'目标')

    def test_record_vectors_are_independent_of_other_records_in_the_batch(self):
        class BatchSensitiveModel(RecordingModel):
            def encode(self,texts,**kwargs):return np.array([[float(len(texts)),float(len(text))] for text in texts])
        encoder=BGEWindowAdapter(SimpleNamespace(model=BatchSensitiveModel(),config=None))
        texts=['长资料'*210,'短事实']
        self.assertEqual(encoder.embed_batch(texts),[encoder.embed(text,'add') for text in texts])

    def test_prompt_source_span_preserves_full_record_and_original_prefix(self):
        text='前段。目标尾部。';start=text.index('目标');end=len(text)
        window={'start':start,'end':end,'text':text[start:end],'unit':'unicode_codepoints','content_sha256':hashlib.sha256(text.encode()).hexdigest()}
        original='请记住，'+text
        item={'content':text,'sources':[{'source_id':'source','evidence':original}],'evidence_window':window}
        content,sources,_=presentation(item)
        span=sources[0]['span']
        self.assertEqual(content,original[span['start']:span['end']])
        self.assertEqual(item['content'],text);self.assertEqual(item['sources'][0]['evidence'],original)
        item['evidence_window']=dict(window,text='伪造片段')
        with self.assertRaises(ValueError):presentation(item)

    def test_quality_metric_does_not_count_an_answer_outside_the_visible_window(self):
        from tools.lcvmem_score import reference_in_visible_body
        text='海岚七号。无关尾段';start=text.index('无关')
        item={'content':text,'sources':[],'evidence_window':{'start':start,'end':len(text),'text':text[start:],
            'unit':'unicode_codepoints','content_sha256':hashlib.sha256(text.encode()).hexdigest()}}
        self.assertFalse(reference_in_visible_body(item,'海岚七号',lambda value:value))
        self.assertTrue(reference_in_visible_body(item,'无关尾段',lambda value:value))
