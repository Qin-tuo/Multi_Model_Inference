"""Real BGE public restore/cutover, current controls, index loss and SIGKILLs."""
import argparse
from contextlib import closing
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.data_lifecycle import check_available
from app.memory_runtime import open_memory
from app.memory_profiles import LEGACY,profile_of,model_path
from app.records import Records
from app.reminders import ReminderService,_deliver_due
from deploy.data.restore import restore
from deploy.memory.migrate import production_profile
from deploy.memory.runtime import load
from tools.project import CACHE,DATA,stamp,write_json


def public(command):
    result=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),*map(str,command)],cwd=ROOT,capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-1800:])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child-phase',choices=['source_fenced','snapshot_staged','snapshot','merged','source_retired'])
    parser.add_argument('--backup',type=Path);parser.add_argument('--source',type=Path);parser.add_argument('--target',type=Path)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--legacy-backup',action='store_true',help='Reindex after backup, then repeatedly recover the older scalar snapshot')
    parser.add_argument('--skip-conversation',action='store_true',help='Data fault checks only; do not generate a model answer')
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/data-restore-smoke.json')
    args=parser.parse_args();service=load(DATA/'qdrant')
    if args.child_phase:
        def kill(phase):
            if phase==args.child_phase:os.kill(os.getpid(),signal.SIGKILL)
        profile=profile_of(json.loads((args.source/'memory/store.json').read_text()))
        restore(args.backup,args.source,args.target,service,production_profile(model_path(profile),profile),checkpoint=kill)
        raise RuntimeError('SIGKILL checkpoint was not reached')
    if not 1<=args.repeats<=10:raise ValueError('Repeats must be1..10')
    checks={};faults=[];error=None;answers=[]
    with tempfile.TemporaryDirectory(prefix='real-data-restore-') as tmp:
        root=Path(tmp);source=root/'source';backup=root/'backup';target=root/'restored'
        try:
            settings={'memory':{'vector_store':'qdrant_server','embedding_profile':LEGACY}} if args.legacy_backup else None
            with open_memory(source/'memory',settings=settings) as memory:
                kept=memory.remember('synthetic-owner','preference','drink','我喜欢喝绿茶。',source_id='seed',request_id='save')['memory_id']
                forgotten=memory.remember('synthetic-owner','preference','old','我收藏旧式水杯。',source_id='old',request_id='old')['memory_id']
                child=memory.remember('synthetic-owner','summary','summary','旧式水杯是我的收藏。',source_id='child',derived_from=(forgotten,),request_id='child')['memory_id']
                guest=memory.remember('synthetic-guest','preference','drink','我喜欢喝咖啡。',source_id='guest',request_id='guest')['memory_id']
                entity=str(uuid.uuid4());memory.sdk.entity_store.insert(vectors=[memory.sdk.embedding_model.embed('旧式水杯','add')],ids=[entity],payloads=[
                    {'data':'旧式水杯','user_id':'synthetic-owner','linked_memory_ids':[forgotten,child]}])
            del memory
            with Records(source/'records.sqlite') as records:
                for mid,turn in ((forgotten,'old'),(child,'child')):
                    records.begin('synthetic-owner','session',turn,'合成旧资料问答');records.attach_memories('synthetic-owner',turn,[mid]);records.finish('synthetic-owner',turn,'合成旧资料回答')
                records.create_note('synthetic-owner','采购','茶叶和杯子',source_id='seed',request_id='note')
            with ReminderService(source/'reminders.sqlite') as reminders:
                due=datetime.now(timezone.utc)+timedelta(hours=1)
                scheduled={name:reminders.create('synthetic-owner','合成'+name,due,request_id=name)['reminder_id'] for name in ('future','cancel','delivered','uncertain')}
            public(['backup-data','--data-dir',source,'--output',backup])
            if args.legacy_backup:public(['reindex-memory','--data-dir',source])
            with Records(source/'records.sqlite') as records:
                records.purge_memory('synthetic-owner',forgotten)
                records.create_note('synthetic-owner','备份后新增','保留这条新笔记',source_id='new',request_id='new-note')
            with open_memory(source/'memory') as memory:
                memory.correct('synthetic-owner',kept,'我喜欢喝桂花茶。',expected_version=1,source_id='correction',request_id='correct')
                memory.forget('synthetic-owner',forgotten,request_id='forget')
                new=memory.remember('synthetic-owner','fact','language','请使用中文交流。',source_id='new',request_id='new')['memory_id']
            del memory
            with ReminderService(source/'reminders.sqlite') as reminders:
                reminders.cancel('synthetic-owner',scheduled['cancel'],request_id='later-cancel')
                for name in ('delivered','uncertain'):
                    _deliver_due(str(source/'reminders.sqlite'),scheduled[name])
                    notice=reminders.claim_next('synthetic-owner')
                    reminders.begin_output(notice['id'],attempt_id=notice['attempt_id'],channel='fixture')
                    if name=='delivered':reminders.acknowledge_delivery(notice['id'],attempt_id=notice['attempt_id'],evidence='fixture-drained')
            restored=json.loads(public(['restore-data','--backup',backup,'--current-data-dir',source,'--data-dir',target]).stdout)
            checks['public_restore_complete']=restored['status']=='PASS' and restored['memories']==3
            with open_memory(target/'memory') as memory:
                drink=memory.search('synthetic-owner',key='drink')['items'][0]
                checks['newer_version_preserved']=drink['memory_id']==kept and drink['version']==2 and drink['content']=='我喜欢喝桂花茶。'
                checks['newer_fact_preserved']=memory.search('synthetic-owner',key='language')['items'][0]['memory_id']==new
                checks['forgotten_and_derived_absent']=not any(item['memory_id'] in {forgotten,child} for item in memory.list_memories('synthetic-owner'))
                checks['guest_isolation']=memory.search('synthetic-guest',key='drink')['items'][0]['memory_id']==guest
                checks['old_entity_erased']=memory.sdk.vector_store.client.retrieve(memory.sdk.collection_name+'_entities',[entity])==[]
                checks['forgotten_history_erased']=memory.sdk.history(forgotten)==[] and memory.sdk.history(child)==[]
            del memory
            with Records(target/'records.sqlite') as records:
                checks['old_transcripts_redacted']=records.recent('synthetic-owner')==[]
                checks['latest_notes_preserved']=len(records.list_notes('synthetic-owner'))==2
            with ReminderService(target/'reminders.sqlite') as reminders:
                statuses={row['id']:row['status'] for row in reminders.list_reminders('synthetic-owner',include_inactive=True)}
                checks['no_duplicate_notification']=reminders.claim_next('synthetic-owner') is None
                checks['uncertain_output_not_replayed']=statuses[scheduled['uncertain']]=='uncertain'
                checks['only_future_reminder_scheduled']=[job.id for job in reminders.scheduler.get_jobs()]==['reminder:'+scheduled['future']]
                checks['job_path_rebound']=reminders.scheduler.get_jobs()[0].args[0]==str(target/'reminders.sqlite')
            for name,callback in [('memory',lambda:open_memory(source/'memory')),('records',lambda:Records(source/'records.sqlite')),('reminders',lambda:ReminderService(source/'reminders.sqlite'))]:
                try:handle=callback()
                except RuntimeError as failure:checks['old_'+name+'_fenced']='retired' in str(failure)
                else:handle.close();checks['old_'+name+'_fenced']=False
            if not args.skip_conversation:
                script=root/'questions.txt';script.write_text('我喜欢喝什么茶？\n/quit\n')
                conversation=public(['interact','--no-vision','--user','synthetic-owner','--data-dir',target,'--input-file',script])
                answers=[line.removeprefix('助手：') for line in conversation.stdout.splitlines() if line.startswith('助手：')]
                checks['public_conversation_uses_corrected_fact']=bool(answers) and '桂花' in answers[0]
            # A fresh verified backup can rebuild an actually missing vector index.
            newer=root/'newer-backup';public(['backup-data','--data-dir',target,'--output',newer])
            binding=json.loads((target/'memory/store.json').read_text())
            with closing(service.client()) as client:client.delete_collection(binding['collection'])
            indexed=root/'index-recovered'
            result=json.loads(public(['restore-data','--backup',newer,'--current-data-dir',target,'--data-dir',indexed]).stdout)
            checks['missing_index_rebuilt']=result['status']=='PASS' and result['memories']==3
            current=indexed
            phases=('snapshot_staged','snapshot','merged','source_retired') if args.legacy_backup else ('source_fenced','merged','source_retired')
            fault_backup=backup if args.legacy_backup else newer
            for phase in phases:
                for iteration in range(args.repeats):
                    destination=root/f'fault-{phase}-{iteration}'
                    killed=subprocess.run([sys.executable,__file__,'--child-phase',phase,'--backup',str(fault_backup),
                        '--source',str(current),'--target',str(destination)],cwd=ROOT,capture_output=True,text=True,timeout=180)
                    case={'phase':phase,'iteration':iteration+1,'actual_sigkill':killed.returncode==-signal.SIGKILL}
                    if not case['actual_sigkill']:raise RuntimeError(killed.stderr[-1800:])
                    for label,path in [('source_fenced',current),('target_fenced',destination)]:
                        try:check_available(path)
                        except RuntimeError:case[label]=True
                        else:case[label]=False
                    resumed=json.loads(public(['restore-data','--backup',fault_backup,'--current-data-dir',current,'--data-dir',destination]).stdout)
                    case['resume_complete']=resumed['status']=='PASS' and resumed['memories']==3
                    check_available(destination)
                    with ReminderService(destination/'reminders.sqlite') as reminders:case['no_duplicate_reminder']=reminders.claim_next('synthetic-owner') is None
                    with open_memory(destination/'memory') as memory:
                        drink=memory.search('synthetic-owner',key='drink')['items'][0]
                        case['current_version_preserved']=drink['memory_id']==kept and drink['version']==2 and drink['content']=='我喜欢喝桂花茶。'
                        case['new_fact_preserved']=memory.search('synthetic-owner',key='language')['items'][0]['memory_id']==new
                        case['forgotten_family_absent']=not any(item['memory_id'] in {forgotten,child} for item in memory.list_memories('synthetic-owner'))
                        case['guest_preserved']=memory.search('synthetic-guest',key='drink')['items'][0]['memory_id']==guest
                        case['erased_history_absent']=memory.sdk.history(forgotten)==[] and memory.sdk.history(child)==[]
                    case['status']='PASS' if all(case[key] for key in ('actual_sigkill','source_fenced','target_fenced','resume_complete','no_duplicate_reminder','current_version_preserved','new_fact_preserved','forgotten_family_absent','guest_preserved','erased_history_absent')) else 'FAIL'
                    faults.append(case);print(json.dumps(case),flush=True);current=destination
            checks['all_recovery_cases_pass']=len(faults)==len(phases)*args.repeats and all(case['status']=='PASS' for case in faults)
        except Exception as failure:error=str(failure);checks['exception']=False
        finally:
            with closing(service.client(timeout=60)) as client:
                collections={}
                for path in root.rglob('store.json'):
                    binding=json.loads(path.read_text());collections[binding['collection']]=binding['store_id']
                for path in root.rglob('data-transition.json'):
                    state=json.loads(path.read_text())
                    if 'binding' in state:collections[state['binding']['collection']]=state['binding']['store_id']
                    for role,name in state.get('staged_snapshots',{}).items():
                        expected=state['binding']['collection']+('_entities' if role=='entities' else '')+'_source'
                        if name!=expected:raise RuntimeError('Unexpected staging collection name')
                        if client.collection_exists(name):
                            metadata=client.get_collection(name).config.metadata or {}
                            if metadata.get('assistant_store_id')!=state['binding']['store_id']:raise RuntimeError('Unrelated staging collection cannot be cleaned')
                            client.delete_collection(name)
                for base,store_id in collections.items():
                    for name in (base+'_entities',base):
                        if not client.collection_exists(name):continue
                        metadata=client.get_collection(name).config.metadata
                        if metadata and metadata.get('assistant_store_id')!=store_id:raise RuntimeError('Unrelated collection cannot be cleaned')
                        client.delete_collection(name)
    report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'answers':answers,'faults':faults,
        'actual_sigkills':sum(case['actual_sigkill'] for case in faults),'repeats_per_phase':args.repeats,
        'legacy_backup':args.legacy_backup,'conversation_tested':not args.skip_conversation,
        'scope':'Actual BGE/Mem0/native Qdrant/SQLite/APScheduler and public backup/restore/interact commands. Synthetic profiles; reminder receipts are explicitly fixture/uncertain, not physical playback. Current SQLite authority preserved, actual vector index loss and restore-process SIGKILL; no entire-disk loss, power loss or complete JT acceptance.'}
    if error:report['error']=error
    write_json(args.output,report)
    print(json.dumps({'status':report['status'],'checks':checks,'actual_sigkills':report['actual_sigkills'],'error':error},ensure_ascii=False),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
