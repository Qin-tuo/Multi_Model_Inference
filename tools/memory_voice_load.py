"""Measure one real voice application sharing its 10k personal memory adapter.

Synthetic WAV input and WAV output; live camera/detection/pose. This is a
development coexistence workload, not acoustic, quality, or final JT acceptance.
Prepare a fresh directory with managed_memory_capacity.py --personal first.
"""
import argparse
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import threading
import time
import wave

from loguru import logger

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.adapters.llama import LlamaConfig
from app.adapters.vision import RosVisionBridge
from app.memory_runtime import open_memory
from app.memory import validate_retrieval_limits
from app.memory_profiles import WINDOWS,HYBRID
from app.records import Records
from app.reminders import ReminderService
from app.voice_assistant import VoiceAssistant
from benchmarks.metrics import sampled_age_summary
from tools.lcvmem_eval import fingerprint
from tools.managed_memory_capacity import workload
from tools.memory_capacity import MemoryMonitor,backend_state,statistics,timed
from tools.project import stamp,write_json,sha256
from tools.voice_assistant_smoke import readback


def vision_summary(node,elapsed):
    return {name:{'received':len(row['arrivals']),'effective_hz':len(row['arrivals'])/elapsed,
        **sampled_age_summary(row['sampled_ages'],row['missing']),
        'invalid_clock':row['invalid_clock'],'class_observations':dict(row['labels'])}
        for name,row in node.rows.items()}


async def main(args):
    logger.remove();logger.add(sys.stderr,level='WARNING')
    runtime=json.loads((ROOT/'configs/runtime.yaml').read_text())
    top_k=args.memory_top_k if args.memory_top_k is not None else runtime['memory']['top_k']
    memory_tokens=args.memory_token_limit if args.memory_token_limit is not None else runtime['memory']['context_token_limit']
    model=LlamaConfig.from_manifest(ROOT/'models/manifest.json')
    validate_retrieval_limits(top_k,memory_tokens,runtime['llama']['context_tokens'],model.max_tokens)
    folder=args.run_dir.resolve();corpus,config,prefix,_=workload(10000,personal=True,profile=args.profile)
    if json.loads((folder/'capacity-run.json').read_text())!=config:raise RuntimeError('Expected the exact personal10k synthetic workload')
    if json.loads((folder/'population.json').read_text())!={'owner':10000,'guest':100}:raise RuntimeError('Finish preparation first')
    if (folder/'clone-receipt.json').exists() and json.loads((folder/'clone-receipt.json').read_text()).get('status')!='COMPLETE':
        raise RuntimeError('Cloned seed is incomplete')
    prepared=json.loads((folder/'prepared.json').read_text())
    started_marker=folder/'coexistence-started.json'
    if args.output.exists() or started_marker.exists():raise RuntimeError('Measurement already attempted; retain its evidence and use a fresh prepared directory')
    prior=json.loads((ROOT/'benchmarks/results/voice-assistant-smoke.json').read_text())
    inputs=[]
    for label,index in [('recall',1),('time',2),('image',3),('arithmetic',5)]:
        path=Path(prior['artifacts'])/f'input-{index}.wav'
        digest=sha256(path)
        if digest!=prior['cases'][index]['input_sha256']:raise RuntimeError('Frozen synthetic voice input changed')
        inputs.append((label,path,digest))
    reader=fingerprint();before=backend_state();artifacts=folder/'coexistence';artifacts.mkdir()
    write_json(started_marker,{'created_at':stamp(),'reader_fingerprint':reader,'trace_stages':args.trace_stages,'debug_loop':args.debug_loop,
        'retrieval_config':{'top_k':top_k,'context_token_limit':memory_tokens},
        'inputs':[{'label':a,'sha256':c} for a,_,c in inputs]})
    queries=[];writes=[];turns=[];lag=[];errors=[];checks={};voice=None;probe=None;bridge=None;thread=None
    observe_stop=threading.Event();voice_done=asyncio.Event();memory_minimum=asyncio.Event();sampling_done=asyncio.Event()
    memory_task=lag_task=None;begin=end=None;vision={};bridge_state={};asr_state={}
    stage_trace=None
    if args.trace_stages:
        from tools.voice_stage_trace import StageTrace
        stage_trace=StageTrace()
    if args.debug_loop:
        loop=asyncio.get_running_loop();loop.set_debug(True);loop.slow_callback_duration=.02
    async def sample_loop():
        loop=asyncio.get_running_loop();target=loop.time()+.02
        while not sampling_done.is_set():
            await asyncio.sleep(max(0,target-loop.time()))
            now=loop.time();delay=max(0,now-target)
            lag.append({'elapsed_ms':delay*1000,'monotonic_ns':time.monotonic_ns(),'missed_periods':int(delay/.02)})
            target+=.02*(int(delay/.02)+1)
    async def memory_load(memory):
        indices=[int((j+.5)*10000/100)//10*10+j%2 for j in range(100)]
        cycle=0;next_cycle=asyncio.get_running_loop().time()
        while not voice_done.is_set():
            for offset in (0,1):
                index=indices[(2*cycle+offset)%len(indices)]
                queued=time.monotonic_ns()
                row,result=await asyncio.to_thread(timed,lambda:memory.search(corpus['owner'],corpus['query_template'].format(index=index)))
                row.update(index=index,long_record=index%10==0,queued_ns=queued,returned_ns=time.monotonic_ns())
                if result is not None:
                    row.update(owner_leaks=sum(x['user_id']!=corpus['owner'] for x in result['items']),
                        expected_key_in_top5=any(x['key']==f'project-{index:05d}' for x in result['items']))
                queries.append(row)
            key=f'coload-write-{cycle}';short=f'合成并发记录T{cycle:04d}要求离开前关闭台灯。'
            body=prefix+short if cycle%2==0 else short;queued=time.monotonic_ns()
            row,result=await asyncio.to_thread(timed,lambda:memory.remember(corpus['owner'],'fact',key,body,source_id=key,request_id=key))
            row.update(index=cycle,long_record=cycle%2==0,queued_ns=queued,returned_ns=time.monotonic_ns())
            if result is not None:
                def verify():
                    found=memory.search(corpus['owner'],key=key)['items']
                    return bool(result['durable'] and len(found)==1 and found[0]['content']==body and memory.sdk.history(result['memory_id'])
                        and memory.operation_status(corpus['owner'],key)['status']=='COMPLETED')
                row['verified_after_timer']=await asyncio.to_thread(verify)
            writes.append(row);cycle+=1
            if cycle>=50:memory_minimum.set()
            next_cycle+=2
            await asyncio.sleep(max(0,next_cycle-asyncio.get_running_loop().time()))
    with MemoryMonitor() as monitor,ExitStack() as stack:
        memory=stack.enter_context(open_memory(folder/'memory'))
        if memory.capabilities()['embedding_profile']!=args.profile:raise RuntimeError('Bound memory index differs from requested workload')
        from qdrant_client import models
        counts={label:memory.sdk.vector_store.client.count(memory.sdk.collection_name,exact=True,count_filter=models.Filter(must=[
            models.FieldCondition(key='user_id',match=models.MatchValue(value=corpus[label])),
            models.FieldCondition(key='namespace',match=models.MatchValue(value='personal'))])).count for label in ('owner','guest')}
        if counts!={'owner':10000,'guest':100}:raise RuntimeError('Prepared personal population changed')
        records=stack.enter_context(Records(folder/'records.sqlite'))
        reminders=stack.enter_context(ReminderService(folder/'reminders.sqlite'))
        try:
            import rclpy
            from tools.vision_runtime_probe import Probe
            bridge=await asyncio.to_thread(lambda:RosVisionBridge.configured().start())
            rclpy.init(domain_id=bridge.domain_id);probe=Probe()
            def observe():
                try:
                    while not observe_stop.is_set():rclpy.spin_once(probe,timeout_sec=.05)
                except Exception as error:errors.append({'stage':'vision_observer','error':str(error)})
            thread=threading.Thread(target=observe,name='coload-vision-probe',daemon=True);thread.start()
            async with asyncio.timeout(25):
                while bridge.status()['state']!='READY' or not all(row['latest'] and time.monotonic_ns()-row['latest']<500_000_000 for row in probe.rows.values()):
                    await asyncio.sleep(.1)
            voice=VoiceAssistant(model,memory,reminders,records,
                user_id=corpus['owner'],vision_frames=bridge.frames,output_path=artifacts/'answers.wav',
                context_tokens=runtime['llama']['context_tokens'],memory_top_k=top_k,memory_context_tokens=memory_tokens)
            if stage_trace:stage_trace.attach(voice,memory)
            await voice.start();begin=time.monotonic_ns();probe.started_ns=begin;probe.measuring=True
            lag_task=asyncio.create_task(sample_loop());memory_task=asyncio.create_task(memory_load(memory))
            # Keep real foreground turns active until at least100 searches and
            #50 durable writes have been attempted during the same interval.
            async with asyncio.timeout(900):
                while len(turns)<args.turns or not memory_minimum.is_set():
                    if memory_task.done():await memory_task;raise RuntimeError('Memory workload ended before foreground input')
                    label,path,digest=inputs[len(turns)%len(inputs)]
                    started=time.monotonic_ns()
                    try:
                        row=await voice.run_file(path,timeout=60)
                        row.update(label=label,input_sha256=digest,started_ns=started,finished_ns=time.monotonic_ns())
                        turns.append(row)
                        print(json.dumps({'turn':len(turns),'label':label,'first_pcm_s':row.get('inferred_speech_end_to_first_pcm_seconds'),
                            'response':row['response']},ensure_ascii=False),flush=True)
                    except Exception as error:
                        turns.append({'label':label,'started_ns':started,'finished_ns':time.monotonic_ns(),'error':str(error)})
                        raise
        except Exception as error:errors.append({'stage':'workload','type':type(error).__name__,'error':str(error)})
        finally:
            voice_done.set()
            if memory_task:
                try:await memory_task
                except Exception as error:errors.append({'stage':'memory','error':str(error)})
            end=time.monotonic_ns();sampling_done.set()
            if lag_task:await lag_task
            if probe:probe.measuring=False
            observe_stop.set()
            if thread:
                await asyncio.to_thread(thread.join,3)
                if thread.is_alive():errors.append({'stage':'close','error':'Vision observer did not stop'})
            if probe:
                if begin:vision=vision_summary(probe,(end-begin)/1e9)
                probe.destroy_node();rclpy.shutdown()
            if bridge:
                bridge_state=bridge.status();await asyncio.to_thread(bridge.close)
            if voice:
                asr_state=voice.stt.status()
                try:await voice.close()
                except Exception as error:errors.append({'stage':'close','error':str(error)})
            if stage_trace:stage_trace.detach()
        # Readbacks are deliberately outside all timed workload intervals.
        if (artifacts/'answers.wav').exists():
            with wave.open(str(artifacts/'answers.wav')) as wav:
                for index,row in enumerate(turns):
                    receipt=row.get('speech_receipt')
                    if not receipt:continue
                    wav.setpos(receipt['first_sample']);pcm=wav.readframes(receipt['bytes_written']//2)
                    path=artifacts/f'answer-{index}.wav'
                    with wave.open(str(path),'wb') as dest:
                        dest.setparams((1,2,16000,0,'NONE','not compressed'));dest.writeframes(pcm)
                    try:row['audio_readback']=await readback(path)
                    except Exception as error:row['readback_error']=str(error)
        qstat,wstat=statistics(queries),statistics(writes)
        foreground_begin=turns[0]['started_ns'] if turns else None
        foreground_end=turns[-1]['finished_ns'] if turns else None
        for row in queries+writes:
            row['foreground_overlap']=bool(turns) and row['queued_ns']<foreground_end and row['returned_ns']>foreground_begin
        overlap_queries=[row for row in queries if row['foreground_overlap']]
        overlap_writes=[row for row in writes if row['foreground_overlap']]
        latencies={label:statistics([{'elapsed_ms':row['inferred_speech_end_to_first_pcm_seconds']*1000} for row in turns
            if row['label']==label and 'inferred_speech_end_to_first_pcm_seconds' in row]) for label,_,_ in inputs}
        checks={'workload_completed':not errors,'minimum_memory_samples':len(queries)>=100 and len(writes)>=50,
            'minimum_overlapping_samples':len(overlap_queries)>=100 and len(overlap_writes)>=50,
            'minimum_voice_turns':len(turns)>=args.turns,'all_voice_outputs':bool(turns) and all(row.get('speech_receipt',{}).get('completion')=='wav_file_flushed' for row in turns),
            'no_voice_errors':voice is not None and not voice.errors,'no_query_errors':qstat['errors']==0,
            'no_write_errors':wstat['errors']==0,'no_owner_leaks':bool(queries) and all(row.get('owner_leaks')==0 for row in queries),
            'durable_writes':bool(writes) and all(row.get('verified_after_timer') for row in writes),
            'query_p95_under200ms':qstat['p95_ms'] is not None and qstat['p95_ms']<=200,
            'write_p95_under500ms':wstat['p95_ms'] is not None and wstat['p95_ms']<=500,
            'file_short_first_pcm_p95_under3s':all(latencies[name]['p95_ms'] is not None and latencies[name]['p95_ms']<=3000 for name in ('recall','time','arithmetic')),
            'file_image_first_pcm_p95_under5s':latencies['image']['p95_ms'] is not None and latencies['image']['p95_ms']<=5000,
            'visual_rates':bool(vision) and all(vision[name]['effective_hz']>=10 for name in ('detection','pose')),
            'visual_age':bool(vision) and all(vision[name]['latest_age_p95_ms'] is not None and vision[name]['latest_age_p95_ms']<=200
                and vision[name]['no_result_fraction']<=.05 and vision[name]['invalid_clock']==0 for name in ('detection','pose')),
            'recall_text_and_audio':any(row['label']=='recall' for row in turns) and all('茉莉' in row.get('response','') and '茉莉' in row.get('audio_readback','') for row in turns if row['label']=='recall'),
            'app_unchanged':reader==fingerprint()}
        report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'errors':errors,
            'reader_fingerprint':reader,'config':config,'initial_population':counts,'run_directory':str(folder),
            'preparation_method':prepared.get('method','sdk-batch-real-bge'),'preparation_sha256':sha256(folder/'prepared.json'),
            'retrieval_config':{'top_k':top_k,'context_token_limit':memory_tokens},
            'workload_begin_ns':begin,'workload_end_ns':end,'query_summary':qstat,'write_summary':wstat,
            'foreground_begin_ns':foreground_begin,'foreground_end_ns':foreground_end,
            'queries_overlapping_foreground':statistics(overlap_queries),'writes_overlapping_foreground':statistics(overlap_writes),
            'query_long':statistics([row for row in queries if row['long_record']]),'query_short':statistics([row for row in queries if not row['long_record']]),
            'write_long':statistics([row for row in writes if row['long_record']]),'write_short':statistics([row for row in writes if not row['long_record']]),
            'voice_latency':latencies,'queries':queries,'writes':writes,'turns':turns,'vision':vision,'bridge':bridge_state,
            'event_loop_lag':{**statistics(lag),'missed20ms_periods':sum(row['missed_periods'] for row in lag)},
            'event_loop_samples':lag,'asr_state':asr_state,'backends_before':before,'backends_after':backend_state(),'resources':monitor.report(),
            'stage_trace':stage_trace.report() if stage_trace else None,'debug_loop':args.debug_loop,
            'scope':'Same real VoiceAssistant and bound int8-window MemoryAdapter;10k personal owner records plus100 guest distractors,10% long owner records. Paced real memory searches/writes overlap repeated frozen synthetic speech inputs,real ASR/LLM/TTS and live RGB-D/detection/pose. File PCM is not physical playback. Periodic visual result age is measured,visual content has no scene ground truth. No reminder/gesture workload,no formal JT,independent quality,B0-B3 or8h acceptance.'}
        write_json(args.output,report)
        print(json.dumps({key:report[key] for key in ('status','checks','query_summary','write_summary','voice_latency','event_loop_lag')},ensure_ascii=False),flush=True)
        return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--turns',type=int,default=20)
    parser.add_argument('--profile',choices=[WINDOWS,HYBRID],default=WINDOWS)
    parser.add_argument('--memory-top-k',type=int,choices=range(1,21))
    parser.add_argument('--memory-token-limit',type=int)
    parser.add_argument('--trace-stages',action='store_true',help='Diagnostic native frames and wall spans; qualify observer overhead before comparisons')
    parser.add_argument('--debug-loop',action='store_true',help='Diagnostic asyncio callbacks slower than20ms; affects measured execution')
    args=parser.parse_args()
    if not 4<=args.turns<=100:parser.error('turns must be4..100')
    raise SystemExit(asyncio.run(main(args)))
