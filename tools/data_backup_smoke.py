"""Real-model backup, upstream snapshot upload, and 10 SIGKILLs per copy phase."""
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
from app.memory_runtime import open_memory
from app.records import Records
from app.reminders import ReminderService
from deploy.data.backup import collection_state,create,sqlite_copy,sqlite_state,upload_snapshot_to_new_collection,validate
from deploy.memory.runtime import load
from tools.project import DATA,stamp,write_json


def public_backup(source,destination):
    result=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'backup-data','--data-dir',str(source),'--output',str(destination)],
        cwd=ROOT,capture_output=True,text=True,timeout=120)
    if result.returncode:raise RuntimeError(result.stderr[-1500:])
    return validate(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child-phase',choices=['database','snapshot','verified'])
    parser.add_argument('--data-dir',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    if args.child_phase:
        def kill(phase):
            if phase==args.child_phase:os.kill(os.getpid(),signal.SIGKILL)
        create(args.data_dir,args.output,checkpoint=kill)
        raise RuntimeError('Requested SIGKILL checkpoint was not reached')
    if not 1<=args.repeats<=10:raise ValueError('Repeats must be1..10')
    service=load(DATA/'qdrant');checks={};faults=[];failure=None;baseline={}
    with tempfile.TemporaryDirectory(prefix='real-data-backup-') as tmp:
        root=Path(tmp);source=root/'application';backup=root/'backup';restored=root/'sqlite-roundtrip';restored.mkdir()
        try:
            with open_memory(source/'memory') as memory:
                fact=memory.remember('synthetic-owner','preference','drink','我喜欢喝桂花茶。',source_id='seed',request_id='save')['memory_id']
                forgotten=memory.remember('synthetic-owner','preference','obsolete','备份后的遗忘样例。',source_id='old')['memory_id']
                memory.remember('synthetic-guest','preference','drink','我喜欢喝咖啡。',source_id='guest')
                memory.sdk.entity_store.insert(vectors=[memory.sdk.embedding_model.embed('饮品','add')],ids=[str(uuid.uuid4())],payloads=[
                    {'data':'饮品','user_id':'synthetic-owner','linked_memory_ids':[fact,forgotten]}])
            del memory
            with Records(source/'records.sqlite') as records:
                records.begin('synthetic-owner','session','turn','合成测试问答')
                records.attach_memories('synthetic-owner','turn',[forgotten])
                records.finish('synthetic-owner','turn','合成测试回答')
                records.create_note('synthetic-owner','备份测试','合成笔记',source_id='turn',request_id='note')
            with ReminderService(source/'reminders.sqlite') as reminders:
                reminder=reminders.create('synthetic-owner','合成提醒',datetime.now(timezone.utc)+timedelta(hours=1),request_id='reminder')['reminder_id']
            manifest=public_backup(source,backup)
            checks['public_cli_complete']=manifest['status']=='COMPLETE'
            verified=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'verify-backup','--backup',str(backup)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            checks['public_verifier_passes']=verified.returncode==0
            checks['all_databases_present']=len(manifest['databases'])==4
            for name,state in manifest['databases'].items():
                sqlite_copy(backup/name,restored/name)
                checks['sqlite_roundtrip:'+name]=sqlite_state(restored/name)==state
            # Later authorized changes establish why a raw old snapshot alone
            # must never become an active restored application.
            with Records(source/'records.sqlite') as records:records.purge_memory('synthetic-owner',forgotten)
            with open_memory(source/'memory') as memory:memory.forget('synthetic-owner',forgotten,request_id='later-forget')
            del memory
            with ReminderService(source/'reminders.sqlite') as reminders:
                reminders.cancel('synthetic-owner',reminder,request_id='later-cancel')
            with closing(service.client(timeout=60)) as client:
                for role,state in manifest['collections'].items():
                    target='backup_probe_'+uuid.uuid4().hex
                    try:
                        upload_snapshot_to_new_collection(client,target,backup/state['file'],manifest['files'][state['file']]['sha256'])
                        checks['native_roundtrip:'+role]=collection_state(client,target)=={key:state[key] for key in ('sha256','points')}
                        if role=='memory':baseline['old_snapshot_still_contains_later_forgotten_id']=bool(client.retrieve(target,[forgotten]))
                    finally:
                        if client.collection_exists(target):client.delete_collection(target)
            with closing(sqlite3.connect(restored/'reminders.sqlite')) as old,closing(sqlite3.connect(source/'reminders.sqlite')) as current:
                baseline['old_snapshot_reminder_active']=old.execute('SELECT status FROM assistant_reminders WHERE id=?',(reminder,)).fetchone()[0]=='active'
                baseline['current_reminder_cancelled']=current.execute('SELECT status FROM assistant_reminders WHERE id=?',(reminder,)).fetchone()[0]=='cancelled'
            baseline.update(status='FAIL',scope='Expected policy baseline: raw faithful old snapshots do not include later deletion/cancellation. No restored application was activated.',
                required_followup='Merge current deletion, transcript redaction, memory versions and delivery/cancellation controls before restoration publication.')
            for phase in ('database','snapshot','verified'):
                for iteration in range(args.repeats):
                    destination=root/f'fault-{phase}-{iteration}'
                    killed=subprocess.run([sys.executable,__file__,'--child-phase',phase,'--data-dir',str(source),'--output',str(destination)],
                        cwd=ROOT,capture_output=True,text=True,timeout=120)
                    case={'phase':phase,'iteration':iteration+1,'actual_sigkill':killed.returncode==-signal.SIGKILL}
                    if not case['actual_sigkill']:raise RuntimeError(killed.stderr[-1500:])
                    try:validate(destination)
                    except RuntimeError:case['partial_backup_rejected']=True
                    else:case['partial_backup_rejected']=False
                    complete=public_backup(source,destination)
                    case['resume_complete']=complete['status']=='COMPLETE'
                    case['server_snapshots_cleaned']=complete['server_snapshots']==[]
                    case['status']='PASS' if all(case[key] for key in ('actual_sigkill','partial_backup_rejected','resume_complete','server_snapshots_cleaned')) else 'FAIL'
                    faults.append(case);print(json.dumps(case),flush=True)
            checks['all_sigkill_recoveries_pass']=len(faults)==3*args.repeats and all(case['status']=='PASS' for case in faults)
        except Exception as error:failure=str(error);checks['exception']=False
        finally:
            path=source/'memory/store.json'
            if path.exists():
                binding=json.loads(path.read_text())
                with closing(service.client(timeout=60)) as client:
                    for name in (binding['collection']+'_entities',binding['collection']):
                        if client.collection_exists(name):
                            metadata=client.get_collection(name).config.metadata
                            if metadata and metadata.get('assistant_store_id')!=binding['store_id']:
                                raise RuntimeError('Unrelated collection cannot be cleaned')
                            client.delete_collection(name)
    report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'faults':faults,
        'actual_sigkills':sum(case['actual_sigkill'] for case in faults),'repeats_per_phase':args.repeats,
        'scope':'Public backup/verify CLI, actual BGE/Mem0/native Qdrant/SQLite, complete snapshot and SQLite roundtrip on isolated synthetic profiles, actual copy-process SIGKILL. Does not implement current-control merge, app cutover or complete restore acceptance.'}
    if failure:report['error']=failure
    write_json(ROOT/'benchmarks/results/data-backup-smoke.json',report)
    if baseline:write_json(ROOT/'benchmarks/results/old-backup-controls-baseline.json',dict(created_at=stamp(),**baseline))
    print(json.dumps({'status':report['status'],'checks':checks,'actual_sigkills':report['actual_sigkills'],'error':failure},ensure_ascii=False),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
