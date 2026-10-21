#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cancel actual local model requests, observe drain, then verify a next request.

Diagnostic evidence only: no physical microphone/speaker or latency acceptance.
Run when no other application is using the same model endpoints.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "01_src"))
import httpx
import numpy as np
from app.audio import wav_pcm
from app.config import Config
from app.llm import LLM
from app.stt import STT
from app.tts import LocalTTS
from app.request_io import TurnCancelled


def wait(predicate, timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate(): return
        time.sleep(.02)
    raise RuntimeError('Timed out waiting for observed native state')


def cancellation(adapter, call, busy):
    if not adapter.idle(): raise RuntimeError('Backend already in use')
    event=threading.Event(); outcome={}
    def run():
        try: call(event); outcome['kind']='completed_before_cancel'
        except TurnCancelled: outcome['kind']='cancelled'
        except Exception as exc: outcome.update(kind='error',error=str(exc))
    thread=threading.Thread(target=run,name='cancel-probe');thread.start()
    try:
        wait(busy,10)
        start=time.monotonic();event.set();adapter.cancel(event)
        thread.join(3)
        outcome['local_return_seconds']=time.monotonic()-start
        if thread.is_alive(): raise RuntimeError('Local cancellation did not return')
        if outcome.get('kind')!='cancelled': raise RuntimeError(f'Cancellation not witnessed: {outcome}')
        outcome['idle_immediately_after_local_return']=adapter.idle()
        wait(adapter.idle)
        outcome['release_confirmed_seconds']=time.monotonic()-start
        outcome['released']=True
        return outcome
    finally:
        event.set();adapter.cancel(event);thread.join(adapter.timeout+2)
        if thread.is_alive(): raise RuntimeError('Probe request has not released')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=Config.load()
    llm=LLM(**{key:value for key,value in asdict(cfg.llm).items() if key!='system_prompt_no_rag'})
    stt=STT(**asdict(cfg.stt))
    tts=LocalTTS(**{key:value for key,value in asdict(cfg.tts).items() if key not in {'first_chunk_chars','max_chunk_chars'}})
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'scope':'Actual local HTTP/native drain and recovery; no physical audio, gesture or performance B0',
            'results':{},'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest()}
    try:
        for adapter in (llm,stt,tts):
            if not adapter.load(): raise RuntimeError(adapter.last_error)
        with httpx.Client(timeout=2,trust_env=False) as client:
            def slot_busy(url): return any(s['is_processing'] for s in client.get(url+'/slots').raise_for_status().json())
            def speech_busy(kind):return client.get(cfg.stt.base_url+'/capabilities').raise_for_status().json()[kind]['concurrency']['active']>0
            report['results']['llm']=cancellation(llm,lambda event:list(llm.generate_stream('请详细列出二十条学习编程的方法，每条说明原因。',cancel_event=event)),lambda:slot_busy(cfg.llm.base_url))
            answer=list(llm.generate_stream('只回答：恢复成功。'))
            assert answer[-1][1].get('done') and any(text for text,_ in answer)
            report['results']['llm']['next_turn_complete']=True
            pcm=wav_pcm(args.input.read_bytes(),cfg.stt.max_seconds)
            audio=np.frombuffer(pcm,dtype='<i2')
            report['results']['asr']=cancellation(stt,lambda event:stt.transcribe(audio,cancel_event=event),lambda:slot_busy(cfg.stt.native_base_url))
            transcript=stt.transcribe(audio)
            assert transcript.get('text') and not transcript.get('error')
            report['results']['asr']['next_turn_complete']=True
            report['results']['tts']=cancellation(tts,lambda event:tts.synthesize('这段语音用于验证停止当前任务后的资源释放情况。取消后不再接收本轮输出，并且等待正在执行的合成步骤结束，再开始下一次对话。',cancel_event=event),lambda:speech_busy('tts'))
            audio=tts.synthesize('恢复成功。')
            assert audio.get('audio') is not None and len(audio['audio'])>0
            report['results']['tts']['next_turn_complete']=True
        report['status']='PASSED_WITH_DECLARED_SCOPE'
    except Exception as exc:
        report.update(status='FAILED',error=str(exc));raise
    finally:
        for adapter in (llm,stt,tts):adapter.unload()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
