"""Cancel the actual voice application during native ASR, LLM and TTS work.

Uses synthetic WAV capture and file playback with real models. Provider replies
are never replaced or held. This does not measure physical barge-in/AEC.
"""
import argparse
import asyncio
from contextlib import ExitStack,closing
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import wave

import httpx
from loguru import logger

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.adapters.llama import LlamaConfig
from app.memory_runtime import open_memory
from app.records import Records
from app.reminders import ReminderService
from app.voice_assistant import VoiceAssistant
from deploy.memory.runtime import load
from deploy import services
from tools.backend_cancellation_probe import snapshot
from tools.project import DATA,stamp,write_json
from tools.voice_assistant_smoke import synthesize,readback
from tools.voice_resource_lifecycle_probe import live


def process_identity(pid):
    try:return {'pid':pid,'start_ticks':(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]}
    except FileNotFoundError:return None


async def terminate_backend(stage):
    component={'asr':'asr','llm':'llama','tts':'speech'}[stage]
    state=await asyncio.to_thread(services.state,[component],health=True)
    if state['backend_status']!='READY':raise RuntimeError('Fault target is not a ready project-owned service')
    before=state['components'][component]
    identities=[process_identity(pid) for pid in before['cgroup_pids']]
    if any(value is None for value in identities):raise RuntimeError('Fault target changed during ownership validation')
    # A signal command can fail after killing only part of the cgroup. Retain
    # identities and always reconcile actual process state before proceeding.
    result=await asyncio.to_thread(services.command,['systemctl','--user','kill','--kill-whom=all','--signal=SIGKILL',services.unit(component)],check=False)
    return {'component':component,'old_processes':identities,'before':before,'signal':'SIGKILL',
        'signal_command_returncode':result.returncode,'signal_command_stderr':result.stderr.strip()}


async def wait_restarted(fault):
    started=time.monotonic()
    async with asyncio.timeout(90):
        while True:
            state=await asyncio.to_thread(services.state,[fault['component']],health=True)
            current=state['components'][fault['component']]
            old_gone=all(process_identity(old['pid'])!=old for old in fault['old_processes'])
            if state['backend_status']=='READY' and old_gone and current['main_pid']!=fault['before']['main_pid']:
                fault['after']=current;fault['ready_wait_seconds']=time.monotonic()-started
                fault['old_processes_exited']=old_gone
                return
            await asyncio.sleep(.2)


async def case(stage,cycle,client,memory,records,reminders,artifacts,fault_mode='cancel'):
    owner=f'synthetic-inflight-{cycle}-{stage}';path=artifacts/f'{cycle}-{stage}.wav'
    voice=VoiceAssistant(LlamaConfig.from_manifest(ROOT/'models/manifest.json'),memory,reminders,records,
        user_id=owner,output_path=path)
    old=current=None;started=time.monotonic();trace=[];row={'cycle':cycle,'stage':stage,'fault':fault_mode,'checks':{}};checks=row['checks'];fault=None
    async def observe():
        state=await snapshot(client,'tts' if stage=='playback' else stage)
        state.update(seconds=time.monotonic()-started,asr_segments=len(voice.stt.pending),samples_written=voice.output.samples_written,
            speech_bytes=voice.speech.current.bytes_written if voice.speech.current else 0)
        trace.append(state);return state
    try:
        await voice.start()
        initial=await observe()
        if initial['busy']:raise RuntimeError('In-flight probe requires initially idle backends')
        old=asyncio.create_task(voice.run_file(artifacts/('remember.wav' if stage=='asr' else 'long-question.wav'),timeout=45))
        async with asyncio.timeout(40):
            while True:
                state=await observe()
                if stage in {'asr','llm'}:active=state.get('slot_busy',False)
                elif stage=='tts':active=voice.speech.current is not None and state['speech_sessions']>0
                else:active=state['speech_bytes']>0
                if active:break
                if old.done():
                    failure=old.exception() if not old.cancelled() else None
                    row['early_error']=str(failure) if failure else None
                    raise RuntimeError('Original turn ended before the required active stage')
                await asyncio.sleep(.02)
        checks['active_stage_observed']=True;checks['original_voice_request_pending']=not old.done()
        row['original_transcript']=voice.session.turn.user_text
        old_token=voice.speech.current.token if voice.speech.current else None
        if fault_mode=='cancel':
            cancelled_at=time.monotonic();await voice.cancel_input('probe_inflight_'+stage)
            row['application_cancel_seconds']=time.monotonic()-cancelled_at
            try:await old;checks['original_caller_cancelled']=False
            except asyncio.CancelledError:checks['original_caller_cancelled']=True
        else:
            fault=row['backend_fault']=await terminate_backend(stage)
            try:
                await asyncio.wait_for(asyncio.shield(old),20);checks['original_caller_failed']=False
            except (Exception,asyncio.CancelledError) as error:
                checks['original_caller_failed']=old.done() and bool(voice.errors)
                row['request_failure']={'type':type(error).__name__,'message':str(error)}
            await wait_restarted(fault)
            checks['backend_restarted_automatically']=True
            checks['old_backend_processes_exited']=fault['old_processes_exited']
        checks['cancelled_turn_not_completed']=not voice.results and not records.recent(owner)
        checks['no_cancelled_memory_write']=not memory.list_memories(owner)
        samples_after_cancel=voice.output.samples_written
        row['samples_after_cancel']=samples_after_cancel
        # Start the next voice request immediately; no provider-idle wait is
        # inserted between cancellation and recovery.
        prior_errors=len(voice.errors)
        recovery_started=time.monotonic();current=asyncio.create_task(voice.run_file(artifacts/'recovery.wav',timeout=45))
        unexpected_pcm=0;saw_recovery_speech=False
        async with asyncio.timeout(50):
            while not current.done():
                attempt=voice.speech.current
                if attempt is not None and attempt.token!=old_token and not attempt.invalidated:saw_recovery_speech=True
                if not saw_recovery_speech:unexpected_pcm=max(unexpected_pcm,voice.output.samples_written-samples_after_cancel)
                await asyncio.sleep(.01)
        result=await current;row['recovery_seconds']=time.monotonic()-recovery_started
        row['response']=result['response'];row['recovery_transcript']=result['transcript'];row['unexpected_pcm_before_new_speech']=unexpected_pcm
        checks.update(no_late_pcm_before_new_speech=unexpected_pcm==0,
            recovered_question='一加一' in result['transcript'],recovered_answer=any(x in result['response'] for x in ('2','二')),
            recovered_audio=result['speech_receipt']['bytes_written']>0,
            only_recovery_completed=len(voice.results)==1 and len(records.recent(owner))==1,
            still_no_memory_side_effect=not memory.list_memories(owner),no_recovery_pipeline_errors=len(voice.errors)==prior_errors)
        if fault_mode=='cancel':checks['no_pipeline_errors']=not voice.errors
        receipt=result['speech_receipt'];clean=artifacts/f'{cycle}-{stage}-recovery.wav'
        with wave.open(str(path)) as wav:
            wav.setpos(receipt['first_sample']);pcm=wav.readframes(receipt['bytes_written']//2)
        with wave.open(str(clean),'wb') as wav:
            wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes(pcm)
        row['recovery_audio_readback']=await readback(clean)
        checks['recovery_wav_recognizes_answer']=any(x in row['recovery_audio_readback'] for x in ('2','二'))
        async with asyncio.timeout(15):
            while (await observe())['busy']:await asyncio.sleep(.02)
        checks['provider_idle_at_case_end']=True
    except Exception as error:
        checks['case_completed']=False;row['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        for task in (old,current):
            if task and not task.done():task.cancel()
            if task:await asyncio.gather(task,return_exceptions=True)
        try:
            await voice.close()
            row['remaining_voice_tasks']=live(voice.runner.task_manager);row['remaining_model_tasks']=live(voice.session.runner.task_manager)
            checks.update(no_tasks_after_close=not row['remaining_voice_tasks'] and not row['remaining_model_tasks'],
                clients_closed=voice.stt._client.is_closed() and voice.tts._http.is_closed and voice.session.bot.llm._client.is_closed(),
                output_closed=voice.output.wav is None and voice.output.stream is None)
        except Exception as error:
            checks['close_completed']=False;row['close_error']={'type':type(error).__name__,'message':str(error)}
        if fault is not None:
            state=await asyncio.to_thread(services.state,[fault['component']],health=True)
            if state['backend_status']!='READY':
                checks['backend_ready_without_probe_repair']=False
                row['manual_service_repair']=await asyncio.to_thread(services.start,[fault['component']],timeout=180)
        row['pipeline_errors']=list(voice.errors);row['asr_counts']=dict(voice.stt.counts)
        row['seconds']=time.monotonic()-started
        raw=artifacts/f'{cycle}-{stage}-observations.json';write_json(raw,trace)
        row['trace']={'path':str(raw),'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'samples':len(trace)}
        row['status']='PASS' if checks and all(checks.values()) else 'FAIL'
    return row


async def main(args):
    logger.remove();logger.add(sys.stderr,level='WARNING')
    source_paths=sorted((ROOT/'app').rglob('*.py'))+[ROOT/'configs/runtime.yaml',ROOT/'models/manifest.json']
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    artifacts=DATA/'fixtures'/('voice-inflight-'+uuid.uuid4().hex);artifacts.mkdir(parents=True)
    for name,text in [('remember','请记住，我喜欢薄荷茶。'),('long-question','请详细介绍太阳系各颗行星的特点。'),('recovery','一加一等于几？请用完整句子回答。')]:
        await synthesize(text,artifacts/(name+'.wav'))
    reports=[];binding={};service=load(DATA/'qdrant')
    with tempfile.TemporaryDirectory(prefix='voice-inflight-') as temporary:
        data=Path(temporary)
        try:
            with ExitStack() as stack:
                memory=stack.enter_context(open_memory(data/'memory'));binding=json.loads((data/'memory/store.json').read_text())
                records=stack.enter_context(Records(data/'records.sqlite'));reminders=stack.enter_context(ReminderService(data/'reminders.sqlite'))
                async with httpx.AsyncClient(timeout=3,trust_env=False) as client:
                    for cycle in range(1,args.cycles+1):
                        for stage in args.stages.split(','):
                            row=await case(stage,cycle,client,memory,records,reminders,artifacts,args.fault);reports.append(row)
                            print(json.dumps({'cycle':cycle,'stage':stage,'status':row['status'],'failed_checks':[k for k,v in row['checks'].items() if not v],
                                'error':row.get('error')},ensure_ascii=False),flush=True)
        finally:
            if binding.get('collection'):
                with closing(service.client()) as client:client.delete_collection(binding['collection'])
    report={'created_at':stamp(),'status':'PASS' if len(reports)==args.cycles*len(args.stages.split(',')) and all(r['status']=='PASS' for r in reports) else 'FAIL',
        'cycles':args.cycles,'stages':args.stages.split(','),'fault':args.fault,'cases':reports,'artifacts':str(artifacts),
        'application_files':source_hashes,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_state':'Working tree identified by application_files; see checkpoint commit containing this report.',
        'scope':'Full VoiceAssistant with synthetic WAV capture/file output and actual Silero/ASR/LLM/TTS/BGE/Mem0/authenticated Qdrant. Cancel or terminate the project-owned backend during observed active stage; same application accepts recovery voice, after automatic backend readiness for crashes. Inspect late PCM and recognize recovery output. No held provider replies, physical transport/AEC or per-kernel timing.'}
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=source_hashes[str(p.relative_to(ROOT))] for p in source_paths):
        report['status']='INVALIDATED';report['reason']='Application source changed during the probe'
    write_json(args.output,report);return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cycles',type=int,default=10)
    parser.add_argument('--fault',choices=['cancel','crash'],default='cancel')
    parser.add_argument('--stages',default='asr,llm,tts,playback');parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/voice-inflight-cancellation.json')
    args=parser.parse_args()
    if not 1<=args.cycles<=100 or any(s not in {'asr','llm','tts','playback'} for s in args.stages.split(',')):parser.error('Invalid cycles/stages')
    if args.fault=='crash' and 'playback' in args.stages.split(','):parser.error('Crash scenarios require --stages asr,llm,tts')
    raise SystemExit(asyncio.run(main(args)))
