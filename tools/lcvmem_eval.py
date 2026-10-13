"""Source-only preparation then real persistent-storage Chinese memory questions."""
import argparse
import asyncio
from contextlib import ExitStack
import hashlib
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.lcvmem_data import load,messages
from tools.lcvmem_prepare import SCORER,sha
from tools.project import stamp,write_json


def fingerprint():
    digest=hashlib.sha256()
    for path in sorted((ROOT/'app').rglob('*.py'))+[ROOT/'configs/runtime.yaml',ROOT/'models/manifest.json']:
        digest.update(str(path.relative_to(ROOT)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def selected(run):
    # Deliberately remove reference answers before entering any runtime phase.
    data={row['id']:{key:row[key] for key in ('id','context','query')} for row in load()}
    return [data[qid] for qid in run['ids']]


def owner_state(memory,user):
    from qdrant_client import models
    condition=models.Filter(must=[models.FieldCondition(key='user_id',match=models.MatchValue(value=user))])
    digest=hashlib.sha256();offset=None;count=0
    while True:
        values,offset=memory.sdk.vector_store.client.scroll(memory.sdk.collection_name,scroll_filter=condition,offset=offset,limit=256,with_payload=True)
        for value in values:digest.update(json.dumps([str(value.id),value.payload],sort_keys=True,ensure_ascii=False).encode());count+=1
        if offset is None:break
    from sqlalchemy import select,func
    with memory.engine.connect() as db:operations=db.execute(select(func.count()).select_from(memory.operations).where(memory.operations.c.user_id==user)).scalar_one()
    return {'facts':count,'sha256':digest.hexdigest(),'operations':operations}


def seed(args,run):
    from app.memory_runtime import open_memory
    from tools.lcvmem_seed import populate
    with open_memory(Path(run.get('memory_directory',args.run_dir/'data/memory'))) as memory:
        for row in selected(run):
            count=populate(memory,run['run_id'],row,source_metadata=run.get('source_metadata','body-only-v1'))
            if owner_state(memory,'lcvmem-'+row['id'])['facts']!=count:raise RuntimeError('Benchmark source import count differs')
            print(json.dumps({'phase':'seed','id':row['id'],'events':count}),flush=True)
    run['seed_complete']=True;run['seed_finished_at']=stamp();write_json(args.run_dir/'lcvmem-run.json',run)


async def answer(args,run):
    from app.adapters.llama import LlamaConfig
    from app.memory_runtime import open_memory
    from app.records import Records
    from app.reminders import ReminderService
    from app.session import AssistantSession
    from loguru import logger
    logger.remove();logger.add(sys.stderr,level='WARNING')
    if not run.get('seed_complete'):raise RuntimeError('Complete the separate seed process before inference')
    current=fingerprint()
    if run.get('reader_fingerprint') not in (None,current):raise RuntimeError('Application changed during this benchmark run; use a new run directory')
    run['reader_fingerprint']=current;run['reader_started_at']=stamp();run['status']='RUNNING';write_json(args.run_dir/'lcvmem-run.json',run)
    (args.run_dir/'answers').mkdir(exist_ok=True)
    config=LlamaConfig.from_manifest(ROOT/'models/manifest.json')
    with open_memory(Path(run.get('memory_directory',args.run_dir/'data/memory'))) as memory:
        for index,row in enumerate(selected(run),1):
            output=args.run_dir/'answers'/(row['id']+'.json')
            if output.exists():continue
            user='lcvmem-'+row['id'];before=owner_state(memory,user);path=args.run_dir/'cases'/row['id']
            if run.get('expected_seed_states') and before!=run['expected_seed_states'][row['id']]:
                raise RuntimeError('Reused benchmark source changed; create a fresh seed run')
            started_path=output.with_suffix('.started.json')
            if started_path.exists():
                prior=json.loads(started_path.read_text())
                write_json(output,{'id':row['id'],'query':row['query'],'response':'','error':{'type':'INTERRUPTED_CASE'},
                    'retrieved':[],'retrieval':{},'calls':[],'side_effect':prior['before']!=before,'seconds':None})
                continue
            write_json(started_path,{'before':before,'started_at':stamp()})
            with ExitStack() as stack:
                records=stack.enter_context(Records(path/'records.sqlite'))
                reminders=stack.enter_context(ReminderService(path/'reminders.sqlite'))
                session=AssistantSession(config,memory,reminders,records,user_id=user)
                result={};error=None;started=time.monotonic()
                try:
                    await session.start()
                    async def emit(_):pass  # Typed-output receipt; no acoustic claim.
                    result=await session.ask(row['query'],emit=emit,timeout=60)
                except Exception as failure:
                    error={'type':type(failure).__name__,'message':str(failure)}
                    if getattr(session,'turn',None):result={'calls':session.turn.calls,'response':''}
                candidates=list(session.turn.memory_candidates) if getattr(session,'turn',None) else []
                try:await session.close()
                except Exception as failure:error=error or {'type':type(failure).__name__,'message':str(failure)}
                after=owner_state(memory,user)
                mutation=before!=after or bool(records.list_notes(user)) or bool(reminders.list_reminders(user,include_inactive=True))
                # IDs bind all retrieved evidence to this benchmark owner.
                cross=False
                for item in candidates:
                    record=memory.sdk.get(item['memory_id'])
                    if record is not None and record.get('user_id')!=user:cross=True
                entry={'id':row['id'],'query':row['query'],**result,'error':error,'retrieved':candidates,
                    'side_effect':mutation,'cross_owner_evidence':cross,'before':before,'after':after,'seconds':time.monotonic()-started}
                write_json(output,entry)
            print(json.dumps({'phase':'answer','completed':index,'total':len(run['ids']),'id':row['id'],
                'error':error['type'] if error else None,'injected':result.get('retrieval',{}).get('injected_count'),
                'side_effect':mutation,'seconds':round(entry['seconds'],2)},ensure_ascii=False),flush=True)
    if fingerprint()!=current:raise RuntimeError('Application changed during benchmark execution')
    run['status']='ANSWERED';run['reader_finished_at']=stamp();write_json(args.run_dir/'lcvmem-run.json',run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['development','heldout','reserve'],default='development')
    parser.add_argument('--limit',type=int);parser.add_argument('--phase',choices=['run','seed','answer'],default='run')
    parser.add_argument('--seed-from',type=Path,help='Development-only comparison using verified unchanged source data and fresh sessions')
    parser.add_argument('--source-metadata',choices=['body-only-v1','calendar-speaker-v1'],default='body-only-v1')
    args=parser.parse_args();args.run_dir=args.run_dir.resolve()
    manifest_path=ROOT/'benchmarks/corpora/lcvmem-v1.json';manifest=json.loads(manifest_path.read_text())
    ids=manifest['partitions'][args.split]['ids']
    if args.limit is not None:
        if args.limit<=0:raise ValueError('Limit must be positive')
        ids=ids[:args.limit]
    args.run_dir.mkdir(parents=True,exist_ok=True,mode=0o700);marker=args.run_dir/'lcvmem-run.json'
    if marker.exists():
        run=json.loads(marker.read_text())
        if (run.get('corpus_sha256')!=sha(manifest_path) or run.get('ids')!=ids or run.get('split')!=args.split
                or run.get('source_metadata','body-only-v1')!=args.source_metadata):
            raise RuntimeError('Existing benchmark ownership/configuration differs')
    else:
        if any(args.run_dir.iterdir()):raise RuntimeError('Benchmark directory is not empty')
        run={'schema_version':1,'run_id':uuid.uuid4().hex,'status':'PREPARING','created_at':stamp(),'split':args.split,'ids':ids,
            'corpus_sha256':sha(manifest_path),'source_metadata':args.source_metadata,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
        if args.seed_from:
            if args.split!='development':raise ValueError('Seed reuse is restricted to development comparisons')
            origin=args.seed_from.resolve();prior=json.loads((origin/'lcvmem-run.json').read_text())
            if (prior.get('status')!='ANSWERED' or prior.get('ids')!=ids or prior.get('corpus_sha256')!=run['corpus_sha256']
                    or prior.get('source_metadata','body-only-v1')!=args.source_metadata):
                raise RuntimeError('Seed source is not a completed matching benchmark')
            expected={}
            for qid in ids:
                result=json.loads((origin/'answers'/(qid+'.json')).read_text())
                if result['before']!=result['after'] or result['side_effect']:raise RuntimeError('Seed run mutated its data')
                expected[qid]=result['before']
            run.update(seed_complete=True,seed_origin=str(origin),expected_seed_states=expected,
                memory_directory=prior.get('memory_directory',str(origin/'data/memory')))
        write_json(marker,run)
    if args.phase in {'seed','answer'}:
        with (args.run_dir/'worker.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.phase=='seed':seed(args,run)
            else:asyncio.run(answer(args,run))
        return 0
    common=['--run-dir',str(args.run_dir),'--output',str(args.output),'--split',args.split,
            '--source-metadata',args.source_metadata]+(['--limit',str(args.limit)] if args.limit else [])
    with (args.run_dir/'orchestrator.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        phases=('answer',) if run.get('seed_complete') else ('seed','answer')
        for phase in phases:
            subprocess.run([sys.executable,__file__,*common,'--phase',phase],cwd=ROOT,check=True)
        subprocess.run([str(SCORER),str(ROOT/'tools/lcvmem_score.py'),'--run-dir',str(args.run_dir),'--output',str(args.output)],cwd=ROOT,check=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
