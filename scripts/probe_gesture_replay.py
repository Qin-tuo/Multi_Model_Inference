#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay locally recorded hand landmarks; optionally exercise real LLM/TTS.

This bypasses camera inference. It verifies temporal-rule reproducibility and
controller routing, never counts as another live gesture or quality sample.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import Config
from app.gestures import GestureRecognizer
from app.llm import LLM
from app.tts import LocalTTS
from app.turn_control import TurnController


def wait(predicate,seconds=30):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.02)
    raise RuntimeError('Replay observation timed out')


def exercise(events,cfg,folder):
    llm=LLM(**{k:v for k,v in asdict(cfg.llm).items() if k!='system_prompt_no_rag'})
    tts=LocalTTS(**{k:v for k,v in asdict(cfg.tts).items() if k not in {'first_chunk_chars','max_chunk_chars'}})
    messages=[];sink=SimpleNamespace(send=lambda msg,**kw:messages.append(msg),ptt_active=False)
    cfg.audio.output_directory=str(folder)
    control=TurnController(cfg,llm,sink,SimpleNamespace(source_epoch='landmark-replay'),tts=tts)
    def rebase(kind):
        original=next(e for e in events if e['kind']==kind)
        shift=time.monotonic()-original['confirmed_monotonic']
        return {**original,**{key:original[key]+shift for key in ['evidence_start_monotonic','evidence_end_monotonic','confirmed_monotonic','expires_monotonic']},
                'source_epoch':'landmark-replay','time_kind':'rebased_recorded_landmark_replay'}
    try:
        for model in (llm,tts):
            if not model.load():raise RuntimeError(model.last_error)
        control.start();wave=control.handle_gesture(rebase('wave'));assert wave['accepted']
        wait(lambda:control.state()['active_turn'] is None)
        greeting=control.state()['last_result'];assert greeting['status']=='complete' and greeting.get('audio_url')
        key=control.submit('请从一数到一百，每个数字独立写一行，不要省略。')['turn_id']
        wait(lambda:any(m.get('turn_id')==key and m['type']=='token' for m in messages))
        stopped=control.handle_gesture(rebase('stop'));assert stopped['accepted']
        wait(lambda:control.state()['active_turn'] is None)
        cancelled=control.state()['last_result'];assert cancelled['status']=='cancelled' and cancelled['resources_released']
        assert control.audio_file(key) is None
        next_key=control.submit('只回答：停止后恢复成功。')['turn_id']
        wait(lambda:control.state()['active_turn'] is None)
        recovered=control.state()['last_result']
        assert recovered['turn_id']==next_key and recovered['status']=='complete' and recovered.get('audio_url')
        assert control.audio_file(greeting['turn_id']) is not None
        return {'scope':'Rebased recorded events through actual local models; not live gestures or physical playback',
                'greeting_complete':True,'stop_cancelled':True,'backend_release_confirmed':True,
                'next_turn_complete_with_audio':True,'earlier_greeting_audio_retained':True,
                'cancelled_turn_audio_unavailable':True}
    finally:control.close();llm.unload();tts.unload()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exercise-models',action='store_true')
    args=parser.parse_args();cfg=Config.load();record=json.loads(args.input.read_text())
    recognizer=GestureRecognizer(cfg.gestures,cfg.perception.hand_threshold);events=[]
    for sample in record['samples']:
        scene=sample['perception'];events+=recognizer.update(scene,now=scene['frame']['received_monotonic']+.001)
    signature=lambda e:(e['kind'],e['first_frame_id'],e['frame_id'])
    expected=[signature(a['event']) for a in record['actions']];actual=[signature(e) for e in events]
    report={'scope':'Recorded-landmark replay, not camera inference','input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'expected_events':expected,'replayed_events':actual,'exact_frame_match':expected==actual}
    assert report['exact_frame_match']
    if args.exercise_models:report['actual_model_routing']=exercise(events,cfg,args.output.parent/'replay-audio')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
