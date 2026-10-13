"""Observe real provider work across cancellation, then verify recovery.

LLM cancellation uses AssistantSession. ASR/TTS exercise the actual OVS HTTP
transport. Slot and admission observations are not per-kernel GPU timestamps.
"""
import argparse
import asyncio
from contextlib import ExitStack,closing
import hashlib
import json
from pathlib import Path
import re
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
from app.session import AssistantSession
from deploy.memory.runtime import load
from tools.project import DATA,stamp,write_json
from tools.voice_assistant_smoke import synthesize


async def snapshot(client,kind):
    result={}
    if kind in {'llm','asr'}:
        response=await client.get(f'http://127.0.0.1:{8088 if kind=="llm" else 8089}/slots')
        response.raise_for_status();slots=response.json()
        if not isinstance(slots,list) or not slots or any(type(s.get('is_processing')) is not bool for s in slots):
            raise RuntimeError('Invalid provider slot observation')
        result['slots']=[{k:s.get(k) for k in ('id','id_task','is_processing','n_prompt_tokens_processed')} for s in slots]
        result['slot_busy']=any(s['is_processing'] for s in slots)
    if kind in {'asr','tts'}:
        response=await client.get('http://127.0.0.1:8621/metrics');response.raise_for_status()
        match=re.search(r'^ovs_sessions_active (\S+)',response.text,re.M)
        if not match:raise RuntimeError('Speech admission metric unavailable')
        result['speech_sessions']=float(match[1])
        if kind=='asr' and result['slot_busy'] and result['speech_sessions']==0:
            # The two endpoints are not an atomic snapshot: decode may finish
            # between reads. Confirm ongoing native work before calling this
            # premature admission release.
            response=await client.get('http://127.0.0.1:8089/slots');response.raise_for_status()
            slots=response.json()
            if not isinstance(slots,list) or not slots or any(type(s.get('is_processing')) is not bool for s in slots):
                raise RuntimeError('Invalid provider slot recheck')
            result['slot_busy_after_admission']=any(s['is_processing'] for s in slots)
    result['busy']=result.get('slot_busy',False) or result.get('speech_sessions',0)>0
    return result


async def exercise(kind,cycle,client,session,records,memory,short_wav,long_wav):
    started=time.monotonic();trace=[];emitted=[];first_pcm=asyncio.Event();received=0;operation=None
    row={'kind':kind,'cycle':cycle,'checks':{}};checks=row['checks'];cancel_at=None
    provider=kind.removesuffix('_timeout');deadline_case=kind.endswith('_timeout')
    async def observe():
        state=await snapshot(client,provider);trace.append({'seconds':time.monotonic()-started,**state});return state
    async def wait_idle(timeout=65):
        async with asyncio.timeout(timeout):
            while (state:=await observe())['busy']:await asyncio.sleep(.02)
        return state
    async def emit(text):emitted.append(text)
    async def tts(text):
        nonlocal received
        async with httpx.AsyncClient(timeout=60,trust_env=False) as transport:
            async with transport.stream('POST','http://127.0.0.1:8621/v1/audio/speech',json={
                'model':'matcha-icefall-zh-en','voice':'0','input':text,'response_format':'pcm'}) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(640):
                    if chunk:received+=len(chunk);first_pcm.set()
    try:
        initial=await observe()
        if initial['busy']:raise RuntimeError('Probe requires idle owned backends; another request is active')
        checks['started_idle']=True
        if provider=='llm':
            operation=asyncio.create_task(session.ask('请详细介绍太阳系各颗行星，逐一说明它们的特点。',emit=emit,timeout=.8 if deadline_case else 90))
        elif provider=='asr':
            operation=asyncio.create_task(client.post('http://127.0.0.1:8621/v1/audio/transcriptions',
                data={'model':'qwen3-asr'},files={'file':('synthetic-long.wav',long_wav,'audio/wav')},
                timeout=httpx.Timeout(65,read=.8) if deadline_case else 65))
        else:
            operation=asyncio.create_task(tts('这是语音取消测试。'+('阳光照在窗前，远处的树叶随风轻轻摇动。'*20)))
        async with asyncio.timeout(20):
            while True:
                active=await observe()
                active_stage=active.get('slot_busy',False) if provider!='tts' else first_pcm.is_set() and active['busy']
                if active_stage:break
                if operation.done():
                    failure=operation.exception() if not operation.cancelled() else None
                    row['early_completion_error']=type(failure).__name__ if failure else None
                    raise RuntimeError('Request finished before an active stage could be observed')
                await asyncio.sleep(.01)
        checks['native_active_stage_observed']=True
        checks['request_pending_at_active_observation']=not operation.done()
        if deadline_case:
            row['injected_deadline_seconds']=.8
            try:
                await operation;checks['caller_timed_out']=False
            except (TimeoutError,httpx.ReadTimeout):checks['caller_timed_out']=True
        else:
            cancel_at=time.monotonic()
            if provider=='llm':await session.cancel()
            else:operation.cancel()
            try:
                await operation;checks['caller_cancelled']=False
            except asyncio.CancelledError:checks['caller_cancelled']=True
            row['caller_cancel_seconds']=time.monotonic()-cancel_at
        caller_exit=time.monotonic()
        after=await observe();row['provider_busy_after_caller_exit']=after['busy']
        await wait_idle();row['provider_idle_observed_seconds_after_caller_exit']=time.monotonic()-caller_exit
        if cancel_at is not None:row['provider_idle_observed_seconds_after_cancel']=time.monotonic()-cancel_at
        checks['provider_idle_before_recovery']=True
        if provider=='asr':
            checks['speech_lease_retained_while_slot_busy']=all(not sample.get('slot_busy') or sample.get('speech_sessions',0)>=1
                or sample.get('slot_busy_after_admission') is False for sample in trace)
            response=await client.post('http://127.0.0.1:8621/v1/audio/transcriptions',data={'model':'qwen3-asr'},
                files={'file':('synthetic-current.wav',short_wav,'audio/wav')},timeout=30)
            response.raise_for_status();row['recovery_transcript']=response.json()['text']
            checks['recovered_asr']='一加一' in row['recovery_transcript']
        elif provider=='tts':
            row['pcm_bytes_received_before_cancel']=received
            response=await client.post('http://127.0.0.1:8621/v1/audio/speech',json={
                'model':'matcha-icefall-zh-en','voice':'0','input':'恢复正常。','response_format':'pcm'},timeout=30)
            response.raise_for_status();row['recovery_pcm_bytes']=len(response.content)
            checks['recovered_tts']=len(response.content)>640 and len(response.content)%2==0
        else:
            checks['cancelled_answer_not_emitted']=not emitted
            checks['cancelled_answer_not_completed_history']=not records.recent(session.user_id)
            checks['no_memory_side_effect']=memory.list_memories(session.user_id)==[]
            result=await session.ask('一加一等于几？',emit=emit);row['recovery_answer']=result['response']
            checks['recovered_llm']=any(x in result['response'] for x in ('2','二'))
            checks['session_admission_restored']=session.backend_available
        await wait_idle(timeout=30)
    except Exception as error:
        checks['completed_without_error']=False;row['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        if operation and not operation.done():
            if provider=='llm':
                try:await session.cancel()
                except Exception:operation.cancel()
            else:operation.cancel()
        if operation:await asyncio.gather(operation,return_exceptions=True)
        try:await wait_idle()
        except Exception as error:
            checks['cleanup_provider_idle']=False;row['cleanup_error']=str(error)
        row['trace']=trace;row['seconds']=time.monotonic()-started
        row['status']='PASS' if all(checks.values()) else 'FAIL'
    return row


async def main(args):
    logger.remove();logger.add(sys.stderr,level='WARNING')
    artifacts=DATA/'fixtures'/('backend-cancel-'+uuid.uuid4().hex);artifacts.mkdir(parents=True)
    await synthesize('一加一等于几？',artifacts/'short.wav')
    with wave.open(str(artifacts/'short.wav')) as wav:pcm=wav.readframes(wav.getnframes())
    with wave.open(str(artifacts/'long.wav'),'wb') as wav:
        wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes((pcm*8)[:16000*2*28])
    short_wav=(artifacts/'short.wav').read_bytes();long_wav=(artifacts/'long.wav').read_bytes()
    binding={};reports=[];service=load(DATA/'qdrant')
    with tempfile.TemporaryDirectory(prefix='backend-cancel-') as temporary:
        data=Path(temporary)
        try:
            with ExitStack() as stack:
                memory=stack.enter_context(open_memory(data/'memory'));binding=json.loads((data/'memory/store.json').read_text())
                records=stack.enter_context(Records(data/'records.sqlite'));reminders=stack.enter_context(ReminderService(data/'reminders.sqlite'))
                async with httpx.AsyncClient(timeout=3,trust_env=False) as client:
                    for cycle in range(1,args.cycles+1):
                        for kind in args.kinds.split(','):
                            session=AssistantSession(LlamaConfig.from_manifest(ROOT/'models/manifest.json'),memory,reminders,records,user_id=f'cancel-{cycle}-{kind}')
                            try:
                                await session.start()
                                row=await exercise(kind,cycle,client,session,records,memory,short_wav,long_wav)
                            finally:await session.close()
                            reports.append(row)
                            print(json.dumps({'kind':kind,'cycle':cycle,'status':row['status'],'failed_checks':[k for k,v in row['checks'].items() if not v],
                                'provider_busy_after_caller_exit':row.get('provider_busy_after_caller_exit'),
                                'idle_seconds_after_caller_exit':row.get('provider_idle_observed_seconds_after_caller_exit')},ensure_ascii=False),flush=True)
        finally:
            if binding.get('collection'):
                with closing(service.client()) as client:client.delete_collection(binding['collection'])
    report={'created_at':stamp(),'status':'PASS' if reports and all(r['status']=='PASS' for r in reports) else 'FAIL',
        'cycles':args.cycles,'cases':reports,'artifacts':str(artifacts),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        'source_state':'working tree; see checkpoint commit containing this report','audio_sha256':hashlib.sha256(long_wav).hexdigest(),
        'scope':'Actual llama LLM/ASR slots and OVS HTTP admission; LLM uses public AssistantSession, ASR/TTS use OVS transport. Cancel or injected short LLM application/ASR HTTP read deadline after observed native work, then provider-idle observation and recovery. Synthetic WAV and TTS text; no per-kernel timing, physical barge-in/AEC, full fault matrix or8h acceptance.'}
    write_json(args.output,report);return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cycles',type=int,default=10)
    parser.add_argument('--kinds',default='llm,llm_timeout,asr,asr_timeout,tts');parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/backend-cancellation.json')
    args=parser.parse_args()
    if not 1<=args.cycles<=100 or any(k not in {'llm','llm_timeout','asr','asr_timeout','tts'} for k in args.kinds.split(',')):parser.error('Invalid cycles/kinds')
    raise SystemExit(asyncio.run(main(args)))
