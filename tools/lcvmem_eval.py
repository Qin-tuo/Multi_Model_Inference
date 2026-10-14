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
from app.context_memory import EVIDENCE_FORMATS, validate_evidence_format


def fingerprint(model_manifest=None):
    digest=hashlib.sha256()
    manifest=Path(model_manifest) if model_manifest is not None else ROOT/'models/manifest.json'
    for path in sorted((ROOT/'app').rglob('*.py'))+[ROOT/'configs/runtime.yaml',manifest]:
        label=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        digest.update(label.encode());digest.update(path.read_bytes())
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
    settings={'memory':{'vector_store':'qdrant_server','embedding_profile':run['requested_profile']}} if run.get('requested_profile') else None
    with open_memory(Path(run.get('memory_directory',args.run_dir/'data/memory')),settings=settings) as memory:
        actual=memory.capabilities()['embedding_profile']
        if run.get('requested_profile') not in (None,actual) or run.get('actual_embedding_profile') not in (None,actual):
            raise RuntimeError('Benchmark bound index differs from its declared profile')
        run['actual_embedding_profile']=actual
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
    model_manifest=Path(run.get('model_manifest_path',ROOT/'models/manifest.json'))
    selected_model=json.loads(model_manifest.read_text())
    entry=selected_model['models']['vlm']
    from tools.project import CACHE
    import httpx
    model_directory=Path(selected_model.get('artifact_directory',CACHE/'models'))
    expected_model=(model_directory/next(f['name'] for f in entry['files'] if not f['name'].startswith('mmproj'))).resolve()
    async with httpx.AsyncClient(timeout=5,trust_env=False) as client:
        response=await client.get('http://127.0.0.1:8088/props');response.raise_for_status();props=response.json()
    if Path(props.get('model_path','')).resolve()!=expected_model or props.get('model_alias')!=entry['server_model_id']:
        raise RuntimeError('Actual native reader differs from selected model manifest')
    run['backend_identity']={key:props.get(key) for key in ('model_path','model_alias','model_ftype','build_info','total_slots')}
    current=fingerprint(model_manifest)
    if run.get('reader_fingerprint') not in (None,current):raise RuntimeError('Application changed during this benchmark run; use a new run directory')
    run['reader_fingerprint']=current;run['reader_started_at']=stamp();run['status']='RUNNING';write_json(args.run_dir/'lcvmem-run.json',run)
    (args.run_dir/'answers').mkdir(exist_ok=True)
    config=LlamaConfig.from_manifest(model_manifest)
    runtime=json.loads((ROOT/'configs/runtime.yaml').read_text());memory_options=runtime['memory']
    memory_top_k=args.memory_top_k if args.memory_top_k is not None else memory_options['top_k']
    memory_tokens=args.memory_token_limit if args.memory_token_limit is not None else memory_options['context_token_limit']
    evidence_format=args.memory_evidence_format or memory_options.get('evidence_format','nested_v1')
    effective={'top_k':memory_top_k,'context_token_limit':memory_tokens,'model_context_tokens':runtime['llama']['context_tokens'],
        'evidence_format':evidence_format}
    if run.get('retrieval_config') not in (None,effective):raise RuntimeError('Retrieval settings changed during benchmark')
    run['retrieval_config']=effective;write_json(args.run_dir/'lcvmem-run.json',run)
    with open_memory(Path(run.get('memory_directory',args.run_dir/'data/memory'))) as memory:
        actual=memory.capabilities()['embedding_profile']
        if run.get('requested_profile') not in (None,actual) or run.get('actual_embedding_profile') not in (None,actual):
            raise RuntimeError('Benchmark index changed between preparation and answering')
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
                session=AssistantSession(config,memory,reminders,records,user_id=user,
                    context_tokens=runtime['llama']['context_tokens'],memory_top_k=memory_top_k,memory_context_tokens=memory_tokens,
                    memory_evidence_format=evidence_format)
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
    if fingerprint(model_manifest)!=current:raise RuntimeError('Application or selected model manifest changed during benchmark execution')
    run['status']='ANSWERED';run['reader_finished_at']=stamp();write_json(args.run_dir/'lcvmem-run.json',run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['development','heldout','reserve'],default='development')
    parser.add_argument('--limit',type=int);parser.add_argument('--phase',choices=['run','seed','answer'],default='run')
    parser.add_argument('--seed-from',type=Path,help='Development-only comparison using verified unchanged source data and fresh sessions')
    parser.add_argument('--source-metadata',choices=['body-only-v1','calendar-speaker-v1'],default='body-only-v1')
    parser.add_argument('--memory-profile',choices=['bge-fp32-single-v1','bge-int8-windows-v1','bge-int8-windows-bm25-v1'])
    parser.add_argument('--memory-top-k',type=int,choices=range(1,21))
    parser.add_argument('--memory-token-limit',type=int)
    parser.add_argument('--memory-evidence-format',choices=EVIDENCE_FORMATS)
    parser.add_argument('--model-manifest',type=Path,help='Explicit candidate identity;does not start or replace a server')
    args=parser.parse_args();args.run_dir=args.run_dir.resolve()
    from app.memory import validate_retrieval_limits
    runtime=json.loads((ROOT/'configs/runtime.yaml').read_text())
    validate_evidence_format(args.memory_evidence_format or runtime['memory'].get('evidence_format','nested_v1'))
    model_manifest=(args.model_manifest or ROOT/'models/manifest.json').resolve()
    model_entry=json.loads(model_manifest.read_text())['models']['vlm']
    if model_entry['backend']!='llama.cpp' or model_entry['context_tokens']!=runtime['llama']['context_tokens']:
        raise ValueError('Selected model must use llama.cpp and the current explicit context budget')
    output_tokens=model_entry['max_output_tokens']
    validate_retrieval_limits(args.memory_top_k if args.memory_top_k is not None else runtime['memory']['top_k'],
        args.memory_token_limit if args.memory_token_limit is not None else runtime['memory']['context_token_limit'],
        runtime['llama']['context_tokens'],output_tokens)
    manifest_path=ROOT/'benchmarks/corpora/lcvmem-v1.json';manifest=json.loads(manifest_path.read_text())
    ids=manifest['partitions'][args.split]['ids']
    if args.limit is not None:
        if args.limit<=0:raise ValueError('Limit must be positive')
        ids=ids[:args.limit]
    args.run_dir.mkdir(parents=True,exist_ok=True,mode=0o700);marker=args.run_dir/'lcvmem-run.json'
    if marker.exists():
        run=json.loads(marker.read_text())
        if (run.get('corpus_sha256')!=sha(manifest_path) or run.get('ids')!=ids or run.get('split')!=args.split
                or run.get('source_metadata','body-only-v1')!=args.source_metadata
                or run.get('requested_profile')!=args.memory_profile
                or run.get('requested_top_k')!=args.memory_top_k or run.get('requested_memory_tokens')!=args.memory_token_limit
                or run.get('requested_memory_evidence_format')!=args.memory_evidence_format
                or run.get('model_manifest_path',str(ROOT/'models/manifest.json'))!=str(model_manifest)
                or run.get('model_manifest_sha256',sha(model_manifest))!=sha(model_manifest)):
            raise RuntimeError('Existing benchmark ownership/configuration differs')
    else:
        if any(args.run_dir.iterdir()):raise RuntimeError('Benchmark directory is not empty')
        run={'schema_version':1,'run_id':uuid.uuid4().hex,'status':'PREPARING','created_at':stamp(),'split':args.split,'ids':ids,
            'corpus_sha256':sha(manifest_path),'source_metadata':args.source_metadata,'requested_profile':args.memory_profile,
            'requested_top_k':args.memory_top_k,'requested_memory_tokens':args.memory_token_limit,
            'requested_memory_evidence_format':args.memory_evidence_format,
            'model_manifest_path':str(model_manifest),'model_manifest_sha256':sha(model_manifest),'model_identity':model_entry,
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
            from app.memory_profiles import profile_of
            actual=profile_of(json.loads((Path(run['memory_directory'])/'store.json').read_text()))
            if args.memory_profile not in (None,actual) or prior.get('actual_embedding_profile') not in (None,actual):
                raise RuntimeError('Reused seed index does not match the requested profile')
            run['actual_embedding_profile']=actual
        write_json(marker,run)
    if args.phase in {'seed','answer'}:
        with (args.run_dir/'worker.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.phase=='seed':seed(args,run)
            else:asyncio.run(answer(args,run))
        return 0
    common=['--run-dir',str(args.run_dir),'--output',str(args.output),'--split',args.split,
            '--source-metadata',args.source_metadata]+(['--limit',str(args.limit)] if args.limit else [])
    if args.memory_profile:common+=['--memory-profile',args.memory_profile]
    if args.memory_top_k is not None:common+=['--memory-top-k',str(args.memory_top_k)]
    if args.memory_token_limit is not None:common+=['--memory-token-limit',str(args.memory_token_limit)]
    if args.memory_evidence_format is not None:common+=['--memory-evidence-format',args.memory_evidence_format]
    if args.model_manifest is not None:common+=['--model-manifest',str(model_manifest)]
    with (args.run_dir/'orchestrator.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        phases=('answer',) if run.get('seed_complete') else ('seed','answer')
        for phase in phases:
            subprocess.run([sys.executable,__file__,*common,'--phase',phase],cwd=ROOT,check=True)
        subprocess.run([str(SCORER),str(ROOT/'tools/lcvmem_score.py'),'--run-dir',str(args.run_dir),'--output',str(args.output)],cwd=ROOT,check=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
