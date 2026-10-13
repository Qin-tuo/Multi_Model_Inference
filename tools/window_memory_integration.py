"""Production adapter, public CLI reindex/restore and real long-memory recall."""
from contextlib import closing
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.context_memory import memory_message
from app.memory_runtime import open_memory
from app.memory_profiles import LEGACY,WINDOWS,profile_of,entity_name
from app.records import Records
from deploy.memory.runtime import load
from tools.long_memory_probe import documents
from tools.project import DATA,stamp,write_json


def command(args):
    result=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),*args],cwd=ROOT,capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return result.stdout


def main(args):
    checks={};queries=[];bindings=[];service=load(DATA/'qdrant');inputs=documents()
    with tempfile.TemporaryDirectory(prefix='window-memory-integration-') as temporary:
        directory=Path(temporary);source=directory/'source';destination=directory/'restored';backup=directory/'backup'
        try:
            with open_memory(source/'memory',settings={'memory':{'vector_store':'qdrant_server','embedding_profile':LEGACY}}) as memory:
                for i,doc in enumerate(inputs):
                    saved=memory.remember('owner','document',doc['key'],doc['content'],source_id=f'synthetic-{i}',source_excerpt='请记住，'+doc['content'],
                        source_date='2026-10-13',source_speaker='user',request_id=f'save-{i}')
                    doc['id']=saved['memory_id']
                memory.remember('guest','document','guest',inputs[0]['content'],source_id='guest',request_id='guest')
            original=json.loads((source/'memory/store.json').read_text());bindings.append(original)
            legacy_backup=directory/'legacy-backup'
            if args.legacy_backup:command(['backup-data','--data-dir',str(source),'--output',str(legacy_backup)])
            migrated=json.loads(command(['reindex-memory','--data-dir',str(source)]))
            checks['public_reindex_completed']=migrated['status']=='PASS'
            current=json.loads((source/'memory/store.json').read_text());bindings.append(current)
            checks['profile_bound']=profile_of(current)==WINDOWS
            with open_memory(source/'memory') as memory:
                for doc in inputs:
                    result=memory.search('owner',f'{doc["project"]}项目的最终交付箱编号是什么？')
                    item=result['items'][0];message,metrics=memory_message([item],'http://127.0.0.1:8088/v1',compact_references=True)
                    queries.append({'project':doc['project'],'correct_id':item['memory_id']==doc['id'],
                        'original_content_intact':item['content']==doc['content'],'window_answer_present':doc['project']+'七号' in item.get('evidence_window',{}).get('text',''),
                        'packed_count':metrics['injected_count'],'tokens':metrics['memory_tokens']})
                checks['all_eight_long_records_recalled']=all(r['correct_id'] and r['original_content_intact'] and r['window_answer_present'] and r['packed_count']==1 for r in queries)
                started=time.monotonic()
                extra=memory.remember('owner','document','measured-extra',inputs[0]['content']+'额外档案。',source_id='extra',request_id='extra')
                write_seconds=time.monotonic()-started
                memory.forget('owner',extra['memory_id'],request_id='remove-extra')
            script=directory/'input.txt';script.write_text('海岚项目的最终交付箱编号是什么？\n/quit\n')
            output=command(['interact','--no-vision','--data-dir',str(source),'--input-file',str(script)])
            answers=[line.removeprefix('助手：') for line in output.splitlines() if line.startswith('助手：')]
            checks['public_recall_after_reindex']=bool(answers) and '海岚七号' in answers[0]
            with open_memory(source/'memory') as memory:
                revised=inputs[0]['content'].replace('海岚七号','海岚九号')
                memory.correct('owner',inputs[0]['id'],revised,expected_version=1,source_id='correction',source_excerpt='更正为：'+revised,request_id='correct',source_speaker='user')
                memory.sdk._upsert_entity('待删除的青竹','PROJECT',inputs[1]['id'],{'user_id':'owner'})
            created=json.loads(command(['backup-data','--data-dir',str(source),'--output',str(backup)]))
            checks['window_backup_completed']=created['status']=='COMPLETE'
            with open_memory(source/'memory') as memory,Records(source/'records.sqlite') as records:
                records.purge_memory('owner',inputs[1]['id'])
                memory.forget('owner',inputs[1]['id'],request_id='forget-after-backup')
            restored=json.loads(command(['restore-data','--backup',str(legacy_backup if args.legacy_backup else backup),'--current-data-dir',str(source),'--data-dir',str(destination)]))
            checks['window_restore_completed']=restored['status']=='PASS'
            restored_binding=json.loads((destination/'memory/store.json').read_text());bindings.append(restored_binding)
            with open_memory(destination/'memory') as memory:
                target=memory.search('owner','海岚项目的交付箱编号是什么？')['items'][0]
                checks['corrected_version_and_id_survive_restore']=target['memory_id']==inputs[0]['id'] and target['version']==2 and '海岚九号' in target['evidence_window']['text']
                checks['forget_not_revived_by_snapshot']=not memory.search('owner',key=inputs[1]['key'])['items']
                checks['guest_separate_after_restore']=len(memory.list_memories('guest'))==1 and len(memory.list_memories('owner'))==7
                entities=memory.sdk.entity_store.list(filters={'user_id':'owner'},top_k=100)[0]
                checks['erased_entity_not_revived']=not any(inputs[1]['id'] in p.payload.get('linked_memory_ids',[]) for p in entities)
            output=command(['interact','--no-vision','--data-dir',str(destination),'--input-file',str(script)])
            final_answers=[line.removeprefix('助手：') for line in output.splitlines() if line.startswith('助手：')]
            checks['public_corrected_recall_after_restore']=bool(final_answers) and '海岚九号' in final_answers[0] and '海岚七号' not in final_answers[0]
            fresh=directory/'fresh';save_script=directory/'save.txt'
            save_script.write_text('请记住，'+inputs[0]['content']+'\n/new\n海岚项目的最终交付箱编号是什么？\n/quit\n')
            output=command(['interact','--no-vision','--data-dir',str(fresh),'--input-file',str(save_script)])
            fresh_answers=[line.removeprefix('助手：') for line in output.splitlines() if line.startswith('助手：')]
            with open_memory(fresh/'memory') as memory:
                from app.tool_policy import literal_fact
                saved=memory.list_memories('owner')
                checks['public_long_save_preserves_literal_fact']=len(saved)==1 and saved[0]['content']==literal_fact('请记住，'+inputs[0]['content'])
                checks['new_profile_default']=memory.capabilities()['embedding_profile']==WINDOWS
            checks['public_long_save_then_new_session_recall']=len(fresh_answers)==2 and '海岚七号' in fresh_answers[1]
            report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'queries':queries,
                'public_answers_before_and_after':[answers,final_answers],'observed_adapter_long_write_seconds':write_seconds,
                'fresh_public_save_answers':fresh_answers,
                'restored_backup_profile':'legacy_before_reindex' if args.legacy_backup else 'current_windows',
                'scope':'Actual production MemoryAdapter/BGE int8/Mem0/authenticated native Qdrant,public reindex/backup/restore/interact with real llama. Synthetic long records; original IDs/versions/owners and deletion authority retained. Not independent language quality,10k latency or full fault acceptance.'}
        except Exception as error:
            checks['completed_without_error']=False
            report={'created_at':stamp(),'status':'FAIL','checks':checks,'queries':queries,'error':{'type':type(error).__name__,'message':str(error)}}
        finally:
            for path in (source/'memory/store.json',destination/'memory/store.json',directory/'fresh/memory/store.json'):
                if path.exists():bindings.append(json.loads(path.read_text()))
            for app in (source,destination):
                for marker in (app/'data-transition.json',app/'memory/reindex-transition.json'):
                    if marker.exists():
                        state=json.loads(marker.read_text())
                        bindings.extend(state[key] for key in ('binding','original_binding','backup_binding') if key in state)
            with closing(service.client()) as client:
                marker=destination/'data-transition.json'
                if marker.exists():
                    state=json.loads(marker.read_text())
                    for name in state.get('staged_snapshots',{}).values():
                        if name in {state['binding']['collection']+'_source',state['binding']['collection']+'_entities_source'} and client.collection_exists(name):client.delete_collection(name)
                for name in {name for binding in bindings for name in [binding['collection'],entity_name(binding['collection'])]}:
                    if client.collection_exists(name):client.delete_collection(name)
    write_json(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--legacy-backup',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/window-memory-integration.json')
    raise SystemExit(main(parser.parse_args()))
