"""Source-only preprocessing and conversation separation, never model quality."""
import json
from pathlib import Path
import unittest
import tempfile

from tools.lcvmem_data import SUFFIX,messages,frozen_split
from tools.lcvmem_prepare import DATA,ROOT


class SourceProtocolTests(unittest.TestCase):
    def test_dates_speakers_and_multiline_content_survive_without_task_footer(self):
        context='以下是2023年04月27日的对话记录：\n“用户: 我喜欢绿茶。\nAI: 你喜欢什么口味？\n清淡还是浓郁？\n”\n\n以下是2023年04月28日的对话记录：\n“用户: 我喜欢清淡口味。\nAI: 记下你的原话。\n”\n\n'+SUFFIX
        result=messages(context)
        self.assertEqual(len(result),4)
        self.assertEqual(result[1]['text'],'你喜欢什么口味？\n清淡还是浓郁？')
        self.assertEqual(result[2]['date'],'2023年04月28日')
        self.assertEqual([x['role'] for x in result],['用户','AI','用户','AI'])
        self.assertNotIn('回答问题',''.join(item['content'] for item in result))

    def test_unrecognized_or_oversized_source_is_not_silently_dropped(self):
        with self.assertRaises(RuntimeError):messages('not the source schema')
        context='以下是2023年04月27日的对话记录：\n“用户: '+('字'*3000)+'\n”\n\n'+SUFFIX
        with self.assertRaisesRegex(RuntimeError,'bound'):messages(context)

    @unittest.skipUnless(DATA.is_file(),'Prepare pinned external CLongEval data')
    def test_frozen_partition_is_reproducible_and_conversation_disjoint(self):
        expected=json.loads((ROOT/'benchmarks/corpora/lcvmem-v1.json').read_text())
        self.assertEqual(frozen_split(),expected)
        ids=[];groups=[]
        for partition in expected['partitions'].values():ids+=partition['ids'];groups+=partition['group_ids']
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(groups),len(set(groups)))
        self.assertGreaterEqual(len(expected['partitions']['heldout']['ids']),200)
        self.assertTrue(set(expected['inspected_example_ids'])<=set(expected['partitions']['development']['ids']))

    def test_fixture_import_matches_sdk_content_and_never_reads_query_or_answer(self):
        from app.memory import Mem0Adapter
        from test_mem0_adapter import actual_sdk
        from tools.lcvmem_seed import populate
        class SourceOnly(dict):
            def __getitem__(self,key):
                if key in {'query','answer'}:raise AssertionError('Oracle/query leakage into source import')
                return super().__getitem__(key)
        context='以下是2023年04月27日的对话记录：\n“用户: 我喜欢绿茶。\nAI: 知道了。\n”\n\n'+SUFFIX
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'data/memory';(root/'lcvmem-run.json').write_text(json.dumps({'run_id':'fixture'}))
            with Mem0Adapter(directory,sdk=actual_sdk(directory),embedding_scope='FIXTURE') as memory:
                count=populate(memory,'fixture',SourceOnly(id='fixture-case',context=context))
                records=memory.list_memories('lcvmem-fixture-case')
                self.assertEqual(count,len(records));self.assertEqual(count,2)
                expected=messages(context)[0]['content']
                normal=memory.remember('normal-owner','event','history-0',expected,source_id='lcv:fixture-case:0',request_id='seed-0')
                copied=next(row for row in records if row['key']=='history-0')
                sdk=memory.search('normal-owner',key='history-0')['items'][0]
                for key in ('content','version','sources','type','key','namespace'):
                    self.assertEqual(copied[key],sdk[key])
                self.assertEqual(populate(memory,'fixture',SourceOnly(id='fixture-case',context=context)),2)
                self.assertEqual(len(memory.list_memories('lcvmem-fixture-case')),2)

    def test_annotated_fixture_matches_public_source_metadata_arguments(self):
        from app.memory import Mem0Adapter
        from test_mem0_adapter import actual_sdk
        from tools.lcvmem_seed import populate
        context='以下是2023年04月27日的对话记录：\n“用户: 我喜欢绿茶。\nAI: 我推荐你尝尝龙井。\n”\n\n'+SUFFIX
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'data/memory';(root/'lcvmem-run.json').write_text(json.dumps({'run_id':'fixture'}))
            with Mem0Adapter(directory,sdk=actual_sdk(directory),embedding_scope='FIXTURE') as memory:
                populate(memory,'fixture',{'id':'annotated','context':context},source_metadata='calendar-speaker-v1')
                copied=memory.search('lcvmem-annotated',key='history-1')['items'][0]
                content=messages(context)[1]['content']
                memory.remember('normal-owner','event','history-1',content,source_id='lcv:annotated:1',request_id='seed-1',
                    source_date='2023-04-27',source_speaker='assistant')
                normal=memory.search('normal-owner',key='history-1')['items'][0]
                self.assertEqual(copied['sources'],normal['sources'])
                self.assertEqual(copied['content'],normal['content'])
