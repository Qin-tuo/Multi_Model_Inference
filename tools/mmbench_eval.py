"""Run sealed Chinese image questions through the actual assistant, then score."""
import argparse
import asyncio
from collections import Counter,defaultdict
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import CACHE,sha256,stamp,write_json

CORPUS=ROOT/'benchmarks/manifests/mmbench-cn-v1.json'
PROTOCOL=ROOT/'benchmarks/manifests/mmbench-evaluation-v1.json'


def inputs(split):
    corpus=json.loads(CORPUS.read_text());part=corpus['partitions'][split]
    path=Path(part['inputs'])
    if sha256(path)!=part['inputs_sha256']:raise RuntimeError('Frozen runtime inputs changed')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if any('answer' in row or 'GT' in row for row in rows):raise RuntimeError('Gold labels leaked into runtime input')
    if [r['id'] for r in rows]!=part['ids']:raise RuntimeError('Input membership/order differs')
    for path,digest in {(r['image_path'],r['image_sha256']) for r in rows}:
        if sha256(Path(path))!=digest:raise RuntimeError('Prepared image changed')
    return corpus,part,rows


def prompt(row):
    text='请根据这张测试图片回答选择题。只输出一个大写选项字母，不要解释。\n'
    if row['hint']:text+='提示：'+row['hint']+'\n'
    return text+'问题：'+row['question']+'\n选项：\n'+''.join(k+'. '+v+'\n' for k,v in row['choices'].items())


def backend_snapshot(candidate_pid=None):
    from deploy import services
    state=services.state(list(services.ORDER),health=True)
    if candidate_pid is None:
        state['evaluation_ready']=state['backend_status']=='READY'
    else:
        others=[name for name in services.ORDER if name!='llama']
        state['candidate']={'pid':candidate_pid,'owns_vlm_port':services.owns_port('llama',[candidate_pid])}
        state['evaluation_ready']=(state['components']['llama']['main_pid']==0
            and state['candidate']['owns_vlm_port'] and all(state['components'][name]['health']['ready'] for name in others))
        state['scope']='Temporary explicit candidate replaces the stopped managedVLM;original unit status retained. Other six managed services must be ready.'
    return state


async def answer(args):
    from PIL import Image
    import httpx
    from loguru import logger
    from app.adapters.llama import LlamaConfig
    from app.memory_runtime import open_memory
    from app.records import Records
    from app.reminders import ReminderService
    from app.session import AssistantSession
    from app.vision_frames import VisionFrameStore
    from tools.lcvmem_eval import fingerprint,owner_state
    from tools.memory_capacity import MemoryMonitor
    from deploy import services

    if args.run_dir.exists() or args.output.exists():raise RuntimeError('Preserve earlier runs;use fresh paths')
    logger.remove();logger.add(sys.stderr,level='WARNING')
    corpus,part,rows=inputs(args.split)
    runtime=json.loads((ROOT/'configs/runtime.yaml').read_text())
    model_manifest=args.model_manifest.resolve()
    manifest=json.loads(model_manifest.read_text());entry=manifest['models']['vlm']
    directory=Path(manifest.get('artifact_directory',CACHE/'models'))
    expected=(directory/next(r['name'] for r in entry['files'] if not r['name'].startswith('mmproj'))).resolve()
    if entry['context_tokens']!=runtime['llama']['context_tokens']:raise RuntimeError('Keep the frozen context capacity')
    candidate_pid=None
    if args.candidate_report:
        parent=json.loads(args.candidate_report.read_text())
        if parent['status']!='RUNNING' or parent['candidate_manifest_sha256']!=sha256(model_manifest):
            raise RuntimeError('Candidate controller manifest/state differs')
        candidate_pid=parent['candidate_pid']
    elif model_manifest!=(ROOT/'models/manifest.json').resolve():
        raise RuntimeError('A candidate requires an explicit controller report and live ownership verification')
    with httpx.Client(timeout=5,trust_env=False) as client:
        response=client.get('http://127.0.0.1:8088/props');response.raise_for_status();props=response.json()
    if Path(props['model_path']).resolve()!=expected or props.get('model_alias')!=entry['server_model_id']:
        raise RuntimeError('Running model differs from selected model')
    before=backend_snapshot(candidate_pid)
    if not before['evaluation_ready']:raise RuntimeError('Evaluation backends/explicit candidate must be ready')
    args.run_dir.mkdir(parents=True,mode=0o700)
    (args.run_dir/'answers').mkdir()
    (args.run_dir/'eval-source.py').write_bytes(Path(__file__).read_bytes())
    identity=fingerprint(model_manifest)
    report={'created_at':stamp(),'status':'RUNNING','split':args.split,'base_questions':100,'variants':len(rows),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':sha256(Path(__file__)),'application_fingerprint':identity,'corpus_sha256':sha256(CORPUS),
        'protocol_sha256':sha256(PROTOCOL),'input_sha256':part['inputs_sha256'],
        'model_identity':{k:props.get(k) for k in ('model_path','model_alias','build_info')},
        'selected_model_manifest':str(model_manifest),'selected_model_manifest_sha256':sha256(model_manifest),
        'active_model_manifest_sha256':sha256(ROOT/'models/manifest.json'),
        'candidate_controller_report':str(args.candidate_report) if args.candidate_report else None,
        'backends_before':before,'completed':0,'errors':0,'unexpected_mutations':0,'image_leaks':0,
        'artifacts':str(args.run_dir.resolve()),'scope':json.loads(PROTOCOL.read_text())['scope']}
    write_json(args.output,report)
    config=LlamaConfig.from_manifest(model_manifest)
    with ExitStack() as stack:
        memory=stack.enter_context(open_memory(args.run_dir/'data/memory',settings=runtime))
        records=stack.enter_context(Records(args.run_dir/'data/records.sqlite'))
        reminders=stack.enter_context(ReminderService(args.run_dir/'data/reminders.sqlite'))
        monitor=stack.enter_context(MemoryMonitor())
        for number,row in enumerate(rows,1):
            gc.collect()
            with Image.open(row['image_path']) as image:rgb=image.convert('RGB').tobytes()
            if len(rgb)!=640*480*3:raise RuntimeError('Prepared image shape differs')
            frames=VisionFrameStore(max_bytes=runtime['vision']['frame_bytes_limit']-16*1024**2,
                retention_seconds=runtime['vision']['frame_window_seconds'],source_kind='synthetic_fixture')
            stop=asyncio.Event();depth=bytes(640*480*2)
            async def feed():
                while not stop.is_set():
                    now=time.monotonic_ns()
                    frames.add(source_time_ns=now,capture_time_ns=now,arrival_time_ns=now,width=640,height=480,
                        rgb_step=1920,rgb_encoding='rgb8',rgb=rgb,depth_step=1280,depth=depth,
                        optical_frame='PUBLIC_DATASET_SYNTHETIC_TRANSPORT')
                    await asyncio.sleep(1/30)
            feeding=asyncio.create_task(feed())
            user='mmbench-'+args.split+'-'+row['id']
            state_before=owner_state(memory,user)
            session=AssistantSession(config,memory,reminders,records,user_id=user,vision_frames=frames,
                context_tokens=runtime['llama']['context_tokens'],memory_top_k=runtime['memory']['top_k'],
                memory_context_tokens=runtime['memory']['context_token_limit'],
                memory_evidence_format=runtime['memory'].get('evidence_format','nested_v1'))
            result={};failure=None;emitted=[];started=time.monotonic()
            async def emit(text):emitted.append(text)
            try:
                await session.start()
                deadline=time.monotonic()+5
                while frames.counts['accepted']<2:
                    if time.monotonic()>deadline:raise TimeoutError('Fixture frame admission')
                    await asyncio.sleep(.01)
                result=await session.ask(prompt(row),emit=emit,visual=True)
            except Exception as error:
                failure={'type':type(error).__name__,'message':str(error)}
                if getattr(session,'turn',None):result={'response':'','calls':session.turn.calls,
                    'prompt_tokens':list(session.budget.measurements),'model_usage':list(session.budget.actual_usage)}
            finally:
                try:await session.close()
                except Exception as error:failure=failure or {'type':type(error).__name__,'message':str(error)}
                stop.set();await feeding
                bot=getattr(session,'bot',None)
                leaked=frames.status()['leased_frames']!=0 or (bot is not None and 'image_url' in str(bot.context.get_messages()))
                frames.reset('case_finished')
            state_after=owner_state(memory,user)
            mutation=state_before!=state_after or bool(records.list_notes(user)) or bool(reminders.list_reminders(user,include_inactive=True))
            measured={'id':row['id'],'base_id':row['base_id'],'category':row['category'],
                'response':result.get('response',''),'error':failure,'calls':result.get('calls',[]),
                'image':result.get('image'),'prompt_tokens':result.get('prompt_tokens'),
                'model_usage':result.get('model_usage'),'seconds':time.monotonic()-started,
                'unexpected_mutation':mutation,'image_leak':leaked,'emitted':bool(emitted),
                'image_sha256':row['image_sha256'],'state_before':state_before,'state_after':state_after}
            write_json(args.run_dir/'answers'/(row['id']+'.json'),measured)
            report['completed']=number;report['errors']+=failure is not None
            report['unexpected_mutations']+=mutation;report['image_leaks']+=leaked
            report['updated_at']=stamp();write_json(args.output,report)
            if number%10==0 or failure:
                print(json.dumps({'completed':number,'total':len(rows),'errors':report['errors'],
                    'unexpected_mutations':report['unexpected_mutations'],'last_id':row['id']}),flush=True)
        report['resources']=monitor.report()
    report['backends_after']=backend_snapshot(candidate_pid)
    checks={'all_variants':report['completed']==len(rows),'application_unchanged':fingerprint(model_manifest)==identity,
        'active_model_manifest_unchanged':sha256(ROOT/'models/manifest.json')==report['active_model_manifest_sha256'],
        'data_unchanged':inputs(args.split)[1]['inputs_sha256']==part['inputs_sha256'],
        'evaluator_unchanged':sha256(Path(__file__))==report['source_sha256'],
        'protocol_unchanged':sha256(PROTOCOL)==report['protocol_sha256'],
        'corpus_manifest_unchanged':sha256(CORPUS)==report['corpus_sha256'],
        'no_mutations':report['unexpected_mutations']==0,'no_image_leaks':report['image_leaks']==0,
        'backends_ready':report['backends_after']['evaluation_ready']}
    report.update(checks=checks,status='ANSWERED' if all(checks.values()) else 'FAIL',completed_at=stamp())
    write_json(args.output,report)
    print(json.dumps({'status':report['status'],'checks':checks,'errors':report['errors']}),flush=True)
    return 0 if report['status']=='ANSWERED' else 1


def score(args):
    from tools.mmbench_scoring import evaluate_group
    if args.output.exists():raise RuntimeError('Preserve earlier score reports')
    run=json.loads(args.run_report.read_text());corpus,part,rows=inputs(run['split'])
    if run['status']!='ANSWERED' or run['completed']!=part['variants']:raise RuntimeError('Complete full inference before scoring')
    reference=Path(part['references'])
    if sha256(reference)!=part['references_sha256']:raise RuntimeError('Reference labels changed')
    gold={r['id']:r['answer'] for r in map(json.loads,reference.read_text().splitlines())}
    groups=defaultdict(list);errors=mutations=leaks=0;artifacts=[]
    for row in rows:
        path=Path(run['artifacts'])/'answers'/(row['id']+'.json');result=json.loads(path.read_text())
        if result['id']!=row['id'] or result['image_sha256']!=row['image_sha256']:raise RuntimeError('Answer/input binding differs')
        errors+=result['error'] is not None;mutations+=result['unexpected_mutation'];leaks+=result['image_leak']
        prediction='' if result['error'] or result['unexpected_mutation'] or result['image_leak'] else result['response']
        groups[row['base_id']].append({'id':row['id'],'base_id':row['base_id'],'category':row['category'],
            'prediction':prediction,'GT':gold[row['id']],**row['choices']})
        artifacts.append({'id':row['id'],'path':str(path),'sha256':sha256(path)})
    questions=[];categories=defaultdict(lambda:Counter(total=0,canonical=0,circular=0));correct=total=unparsed=strict=0
    for base_id,group in groups.items():
        evaluated=evaluate_group(group)
        index=next(i for i,row in enumerate(group) if row['id']==base_id)
        canonical=evaluated['variant_correct'][index];circular=evaluated['circular_correct']
        category=group[0]['category'];categories[category].update(total=1,canonical=int(canonical),circular=int(circular))
        correct+=sum(evaluated['variant_correct']);total+=len(group);unparsed+=sum(x=='Z' for x in evaluated['parsed'])
        strict+=sum(re.fullmatch('[ABCD]',row['prediction'].strip()) is not None for row in group)
        questions.append({'base_id':base_id,'category':category,'variants':len(group),
            'canonical_correct':canonical,'circular_correct':circular,'parsed':evaluated['parsed']})
    primary=sum(r['canonical_correct'] for r in questions)/len(questions)
    report={'created_at':stamp(),'status':'DEVELOPMENT_RESULT' if run['split']=='development' else 'HELDOUT_RESULT',
        'split':run['split'],'base_questions':len(questions),'variants':total,'canonical_accuracy':primary,
        'circular_accuracy':sum(r['circular_correct'] for r in questions)/len(questions),
        'variant_accuracy':correct/total,'unparsed':unparsed,'strict_letter_outputs':strict,
        'screening_target_80_percent_met':primary>=.8,'errors':errors,'unexpected_mutations':mutations,'image_leaks':leaks,
        'categories':dict(categories),'questions':questions,'answer_artifacts':artifacts,
        'inference_report':str(args.run_report),'inference_report_sha256':sha256(args.run_report),
        'corpus_sha256':sha256(CORPUS),'scorer_lock_sha256':sha256(ROOT/'deploy/evaluation/mmbench.lock.json'),
        'protocol_sha256':sha256(PROTOCOL),'scope':run['scope']}
    write_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('status','canonical_accuracy','circular_accuracy','variant_accuracy','unparsed','errors','unexpected_mutations','screening_target_80_percent_met')}),flush=True)
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    run=sub.add_parser('answer');run.add_argument('--run-dir',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--split',choices=['development','heldout','reserve'],default='development')
    run.add_argument('--model-manifest',type=Path,default=ROOT/'models/manifest.json')
    run.add_argument('--candidate-report',type=Path)
    scoring=sub.add_parser('score');scoring.add_argument('--run-report',type=Path,required=True);scoring.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(asyncio.run(answer(args)) if args.action=='answer' else score(args))
