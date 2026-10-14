"""Real application regression for a memory-topic question vs explicit listing."""
import argparse
import asyncio
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from loguru import logger
import httpx

from app.adapters.llama import LlamaConfig
from app.memory_runtime import open_memory
from app.records import Records
from app.reminders import ReminderService
from app.session import AssistantSession
from tools.lcvmem_eval import selected,owner_state,fingerprint
from tools.project import CACHE,stamp,sha256,write_json


async def main(args):
    logger.remove();logger.add(sys.stderr,level='WARNING')
    if args.run_dir.exists() or args.output.exists():raise RuntimeError('Use fresh regression paths')
    source=json.loads((args.source_run/'lcvmem-run.json').read_text())
    if source['status']!='ANSWERED' or source['split']!='development':raise RuntimeError('Use completed development data only')
    row=next(row for row in selected(source) if row['id']==args.case_id)
    prior=json.loads((args.source_run/'answers'/(args.case_id+'.json')).read_text())
    if prior['before']!=prior['after']:raise RuntimeError('Source run mutated the fixture')
    manifest=json.loads(args.manifest.read_text());entry=manifest['models']['vlm']
    model_directory=Path(manifest.get('artifact_directory',CACHE/'models'))
    model_path=model_directory/next(f['name'] for f in entry['files'] if not f['name'].startswith('mmproj'))
    async with httpx.AsyncClient(timeout=5,trust_env=False) as client:
        response=await client.get('http://127.0.0.1:8088/props');response.raise_for_status();props=response.json()
    if Path(props['model_path']).resolve()!=model_path.resolve():raise RuntimeError('Wrong native reader for regression')
    args.run_dir.mkdir(parents=True,mode=0o700)
    runtime=json.loads((ROOT/'configs/runtime.yaml').read_text())
    user='lcvmem-'+args.case_id;results=[];errors=[];before=None;after=None
    with ExitStack() as stack:
        memory=stack.enter_context(open_memory(Path(source.get('memory_directory',args.source_run/'data/memory'))))
        before=owner_state(memory,user)
        if before!=prior['before']:raise RuntimeError('Prepared source facts changed')
        records=stack.enter_context(Records(args.run_dir/'records.sqlite'))
        reminders=stack.enter_context(ReminderService(args.run_dir/'reminders.sqlite'))
        session=AssistantSession(LlamaConfig.from_manifest(args.manifest),memory,reminders,records,user_id=user,
            context_tokens=runtime['llama']['context_tokens'],memory_top_k=8,memory_context_tokens=768,
            memory_evidence_format='flat_v2')
        try:
            await session.start()
            async def emit(_):pass
            for kind,text in [('topic',row['query']),('inventory','列出我的记忆。')]:
                result=await session.ask(text,emit=emit)
                results.append({'kind':kind,**result})
        except Exception as error:errors.append({'type':type(error).__name__,'error':str(error)})
        finally:await session.close()
        after=owner_state(memory,user)
    write_json(args.run_dir/'raw-turns.json',results)
    checks={'two_complete_turns':len(results)==2,'no_errors':not errors,'source_unchanged':before==after}
    if len(results)==2:
        checks.update(topic_uses_retrieval=results[0]['retrieval']['injected_count']>0,
            topic_no_inventory=not any(c['tool']=='list_memories' for c in results[0]['calls']),
            topic_target_answer=args.expected in results[0]['response'],
            inventory_called_once=sum(c['tool']=='list_memories' and c['outcome'].get('ok') for c in results[1]['calls'])==1)
    report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'errors':errors,
        'case_id':args.case_id,'reader_fingerprint':fingerprint(args.manifest.resolve()),
        'model_manifest_sha256':sha256(args.manifest),'source_sha256':sha256(Path(__file__)),
        'turns':[{'kind':r['kind'],'response_sha256':hashlib.sha256(r['response'].encode()).hexdigest(),
            'tools':[{'name':c['tool'],'ok':c['outcome'].get('ok'),'code':c['outcome'].get('code')} for c in r['calls']],
            'prompt_tokens':r['prompt_tokens'],'seconds':r['elapsed_seconds']} for r in results],
        'artifacts':str(args.run_dir.resolve()),'scope':'One existing development question plus explicit inventory through real AssistantSession/BGE/nativeLLM andunchanged source store. Target checked only after inference. Raw corpus/answers outside repository. Regression only,not a new full development score orindependent quality/JT acceptance.'}
    write_json(args.output,report);print(json.dumps({'status':report['status'],'checks':checks,'errors':errors}),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--case-id',required=True)
    parser.add_argument('--expected',required=True,help='Evaluator-only answer fragment;never passed to model')
    parser.add_argument('--manifest',type=Path,default=ROOT/'models/manifest.json')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
