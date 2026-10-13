"""Repeated real speech inference with held ASR callbacks and native teardown.

The provider completes real recognition before the callback is held. This
qualifies application task/connection cleanup, not GPU preemption or acoustics.
"""
import argparse
import asyncio
from contextlib import ExitStack,closing
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid

from loguru import logger

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.adapters.llama import LlamaConfig
from app.memory_runtime import open_memory
from app.records import Records
from app.reminders import ReminderService
from app.voice_assistant import VoiceAssistant
from deploy.memory.runtime import load
from tools.project import DATA,stamp,write_json
from tools.voice_assistant_smoke import synthesize
from tools.voice_asr_interruption_probe import feed,wait_for


def live(manager):return [task.get_name() for task in manager.current_tasks() if not task.done()]


async def main(args):
    logger.remove();logger.add(sys.stderr,level='WARNING')
    artifacts=DATA/'fixtures'/('voice-resource-'+uuid.uuid4().hex);artifacts.mkdir(parents=True)
    for name,text in [('cancelled','请记住，我喜欢薄荷茶。'),('current','一加一等于几？')]:
        await synthesize(text,artifacts/(name+'.wav'))
    reports=[];service=load(DATA/'qdrant');binding={}
    with tempfile.TemporaryDirectory(prefix='voice-resource-') as temporary:
        data=Path(temporary)
        try:
            with ExitStack() as stack:
                memory=stack.enter_context(open_memory(data/'memory'))
                binding=json.loads((data/'memory/store.json').read_text())
                records=stack.enter_context(Records(data/'records.sqlite'))
                reminders=stack.enter_context(ReminderService(data/'reminders.sqlite'))
                config=LlamaConfig.from_manifest(ROOT/'models/manifest.json')
                for index in range(args.cycles):
                    for phase in ('cancel_and_recover','close_with_pending_callback'):
                        owner=f'synthetic-resource-{index}-{phase}';entered=asyncio.Event();release=asyncio.Event();transcripts=[]
                        baseline_tasks=set(asyncio.all_tasks())
                        voice=VoiceAssistant(config,memory,reminders,records,user_id=owner,
                            output_path=artifacts/f'{index}-{phase}.wav')
                        actual=voice.stt._transcribe
                        async def delayed(audio):
                            response=await actual(audio);transcripts.append(response.text)
                            if len(transcripts)==1:
                                entered.set();await release.wait()
                            return response
                        voice.stt._transcribe=delayed
                        row={'cycle':index+1,'phase':phase,'checks':{}};checks=row['checks'];start=time.monotonic()
                        try:
                            await voice.start()
                            await feed(voice,artifacts/'cancelled.wav')
                            await asyncio.wait_for(entered.wait(),15)
                            checks['actual_asr_completed_before_injected_delay']='薄荷' in transcripts[0]
                            checks['no_model_request_from_held_callback']=voice.answer_task is None and not voice.results
                            if phase=='cancel_and_recover':
                                await voice.cancel_input('resource_probe_cancel_before_callback')
                                checks['cancel_cleared_stop_timers']=not any(name.startswith(str(voice.stop_strategy)+'::') for name in live(voice.runner.task_manager))
                                release.set();await wait_for(lambda:not voice.stt.pending)
                                result=await voice.run_file(artifacts/'current.wav',timeout=45)
                                row['response']=result['response'];row['recovery_transcript']=result['transcript']
                                checks.update(recovered_answer=any(word in result['response'] for word in ('2','二')),
                                    recovery_only='一加一' in result['transcript'] and '薄荷' not in result['transcript'],
                                    output_received=result['speech_receipt']['bytes_written']>0,
                                    stale_callback_discarded=voice.stt.counts['obsolete_results_discarded']==1)
                            checks['cancelled_preference_not_saved']=memory.list_memories(owner)==[]
                            checks['cancelled_text_absent_from_completed_history']='薄荷' not in json.dumps(records.recent(owner),ensure_ascii=False)
                            checks['no_pipeline_errors']=not voice.errors
                        except Exception as error:
                            checks['scenario_completed']=False;row['error']={'type':type(error).__name__,'message':str(error)}
                        finally:
                            closing_started=time.monotonic()
                            try:
                                await voice.close()
                                row['close_seconds']=time.monotonic()-closing_started
                                row['remaining_voice_tasks']=live(voice.runner.task_manager)
                                row['remaining_model_tasks']=live(voice.session.runner.task_manager)
                                row['extra_live_async_tasks']=[t.get_name() for t in asyncio.all_tasks() if t not in baseline_tasks and not t.done()]
                                checks.update(voice_tasks_joined=not row['remaining_voice_tasks'],
                                    model_tasks_joined=not row['remaining_model_tasks'],
                                    no_extra_async_tasks=not row['extra_live_async_tasks'],
                                    stt_client_closed=voice.stt._client.is_closed(),
                                    tts_client_closed=voice.tts._http.is_closed,
                                    llm_client_closed=voice.session.bot.llm._client.is_closed(),
                                    budget_client_closed=voice.session.budget.client.is_closed,
                                    output_file_closed=voice.output.wav is None and voice.output.stream is None,
                                    asr_segments_released=not voice.stt.pending and voice.stt.status()['queued_segments']==0)
                                if phase=='close_with_pending_callback':checks['close_did_not_emit_held_text']=not voice.results and voice.answer_task is None
                            except Exception as error:
                                checks['close_completed']=False;row['close_error']={'type':type(error).__name__,'message':str(error)}
                            # Probe cleanup follows measurements and cannot turn
                            # failed native cleanup into a successful result.
                            release.set()
                            for manager in (voice.runner.task_manager,voice.session.runner.task_manager):
                                for task in list(manager.current_tasks()):
                                    if not task.done():await manager.cancel_task(task)
                            await voice.session.bot.llm._client.close()
                        row['actual_asr_transcripts']=transcripts;row['seconds']=time.monotonic()-start
                        row['process_fd_count']=len(list(Path('/proc/self/fd').iterdir()))
                        row['status']='PASS' if all(checks.values()) else 'FAIL';reports.append(row)
                        print(json.dumps({'cycle':index+1,'phase':phase,'status':row['status'],
                            'failed_checks':[k for k,v in checks.items() if not v]},ensure_ascii=False),flush=True)
        finally:
            if binding.get('collection'):
                with closing(service.client()) as client:client.delete_collection(binding['collection'])
    report={'created_at':stamp(),'status':'PASS' if len(reports)==2*args.cycles and all(r['status']=='PASS' for r in reports) else 'FAIL',
        'cycles':args.cycles,'cases':reports,'artifacts':str(artifacts),
        'scope':'Synthetic WAV input; actual Silero/OVS ASR/llama/BGE/Mem0/authenticated native Qdrant/TTS. ASR response held after provider completion. Measures application cancellation/recovery/teardown tasks, owned HTTP clients and file handles. Not physical AEC/interruption, GPU preemption, full fault matrix or8h stability acceptance.'}
    write_json(args.output,report);return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cycles',type=int,default=10)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/voice-resource-lifecycle.json')
    args=parser.parse_args()
    if not 1<=args.cycles<=100:parser.error('cycles must be1..100')
    raise SystemExit(asyncio.run(main(args)))
