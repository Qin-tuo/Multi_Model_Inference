#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify generated Chinese WAV -> ASR -> session -> LLM -> TTS on an empty app."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from app.audio import pcm_wav
from app.config import Config
from app.tts import LocalTTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8093');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=Config.load();report={'scope':'Generated Chinese WAV and actual local ASR/LLM/TTS; not microphone or physical speaker'}
    tts=LocalTTS(**{k:v for k,v in asdict(cfg.tts).items() if k not in {'first_chunk_chars','max_chunk_chars'}})
    try:
        with httpx.Client(base_url=args.base_url,timeout=5,trust_env=False) as client:
            def state():return client.get('/api/state').raise_for_status().json()
            def result(response):
                key=response.raise_for_status().json()['turn_id'];end=time.monotonic()+90
                while time.monotonic()<end:
                    current=state()
                    if current['active_turn'] is None and current['last_result'] and current['last_result']['turn_id']==key:
                        value=current['last_result'];assert value['status']=='complete',value
                        reply=client.get(value['audio_url']).raise_for_status().content
                        value['download_sha256']=hashlib.sha256(reply).hexdigest()
                        return value
                    time.sleep(.05)
                raise RuntimeError('Speech session observation timed out')
            initial=state()
            if initial['active_turn'] or initial['session']['retained_turns'] or not initial['speech_ready']:
                raise RuntimeError('Use an isolated empty app with file speech enabled')
            assert tts.load(),tts.last_error
            spoken=tts.synthesize('我最喜欢的颜色是蓝色，请你记住。')
            assert spoken.get('audio') is not None,spoken
            wav=pcm_wav(spoken['audio'].tobytes());report['input_sha256']=hashlib.sha256(wav).hexdigest()
            report['first']=result(client.post('/api/audio',content=wav,headers={'Content-Type':'audio/wav'}))
            assert '蓝色' in report['first']['prompt']
            assert not report['first'].get('actions_committed'), 'Ordinary session recall must not create a durable note'
            report['recall']=result(client.post('/api/chat',json={'text':'我最喜欢的颜色是什么？只回答颜色，没有提供就回答不知道。'}))
            assert '蓝色' in report['recall']['text']
            client.post('/api/session/clear',json={'session_id':state()['session']['id']}).raise_for_status()
            report['after_clear']=result(client.post('/api/chat',json={'text':'我最喜欢的颜色是什么？只回答颜色，没有提供就回答不知道。'}))
            assert '蓝色' not in report['after_clear']['text'] and '不知道' in report['after_clear']['text']
            report['status']='PASSED_WITH_DECLARED_SCOPE'
    except Exception as exc:
        report.update(status='FAILED',error=str(exc));raise
    finally:
        tts.unload();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'asr':report['first']['prompt'],'recall':report['recall']['text'],
                      'after_clear':report['after_clear']['text']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
