"""Exercise native cache eviction with synthetic text and padded file audio.

Not a model-quality or latency benchmark. Input is an existing synthetic TTS
fixture,never microphone capture; both configured llama.cpp services stay local.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import re
import sys
import time
import wave

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy import services
from tools.memory_capacity import MemoryMonitor
from tools.project import DATA,stamp,write_json,sha256


def logs(component):return list((DATA/'logs/backends').glob(component+'.log*'))


def positions(component):
    return {(p.stat().st_dev,p.stat().st_ino):p.stat().st_size for p in logs(component)}


def cache_evidence(component,before):
    lines=[]
    for path in logs(component):
        stat=path.stat()
        with path.open('rb') as stream:
            stream.seek(before.get((stat.st_dev,stat.st_ino),0));lines+=stream.read().decode(errors='replace').splitlines()
    states=[]
    for line in lines:
        match=re.search(r'cache state: (\d+) prompts, ([\d.]+) MiB \(limits: ([\d.]+) MiB',line)
        if match:states.append({'prompts':int(match[1]),'mib':float(match[2]),'limit_mib':float(match[3])})
    return {'states':states,'evictions':sum('removing oldest entry' in line for line in lines),
        'maximum_logged_mib':max((s['mib'] for s in states),default=None)}


def main(args):
    if args.output.exists():raise RuntimeError('Keep the previous cache report; use a new output')
    settings=json.loads((ROOT/'configs/runtime.yaml').read_text())
    limits={'llama':settings['llama']['prompt_cache_ram_mib'],'asr':settings['speech']['asr_prompt_cache_ram_mib']}
    before=services.state(['llama','asr','speech'],health=True)
    if before['backend_status']!='READY':raise RuntimeError('All speech/model backends must be ready')
    commands={}
    for name in limits:
        candidates=[]
        for pid in before['components'][name]['cgroup_pids']:
            command=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
            if Path(command[0]).name=='llama-server':candidates.append((pid,command))
        if len(candidates)!=1:raise RuntimeError('Could not identify the owned llama-server')
        pid,command=candidates[0]
        if command[command.index('--cache-ram')+1]!=str(limits[name]) or command[command.index('-c')+1]!='2048':
            raise RuntimeError('Actual context/cache command differs from the probe')
        commands[name]={'pid':pid,'cache_ram_mib':limits[name],'context_tokens':2048}
    prior=json.loads((ROOT/'benchmarks/results/voice-assistant-smoke.json').read_text())
    audio=Path(prior['artifacts'])/'input-1.wav'
    if sha256(audio)!=prior['cases'][1]['input_sha256']:raise RuntimeError('Synthetic input changed')
    with wave.open(str(audio)) as wav:
        if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(16000,1,2):raise RuntimeError('Unexpected fixture format')
        pcm=wav.readframes(wav.getnframes())
    offsets={name:positions(name) for name in limits};text_rows=[];audio_rows=[]
    with MemoryMonitor() as monitor,httpx.Client(timeout=60,trust_env=False) as client:
        padding='这是一份只用于本地缓存边界检查的合成资料，不包含用户信息。'*22
        for i in range(20):
            tag=f'K{i:02d}';prompt=f'合成轮次{i}。'+padding+f'\n请只回复本轮校验标签{tag}。'
            body={'model':'assistant-vlm','messages':[{'role':'system','content':'直接逐字回复用户要求的校验标签，不添加其他文字。'},
                {'role':'user','content':prompt}],'temperature':0,'max_tokens':8,'cache_prompt':True}
            response=client.post('http://127.0.0.1:8088/v1/chat/completions',json=body);response.raise_for_status();data=response.json()
            answer=data['choices'][0]['message'].get('content','')
            text_rows.append({'case':i,'expected_tag':tag,'response':answer,'tag_present':tag in answer,'usage':data.get('usage')})
        for i in range(20):
            buffer=io.BytesIO()
            with wave.open(buffer,'wb') as wav:
                wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes(bytes(3200*i)+pcm)
            response=client.post('http://127.0.0.1:8621/v1/audio/transcriptions',data={'model':'qwen3-asr'},
                files={'file':(f'synthetic-padding-{i}.wav',buffer.getvalue(),'audio/wav')})
            response.raise_for_status();text=response.json()['text']
            audio_rows.append({'case':i,'leading_silence_seconds':i*.1,'transcript':text,'question_preserved':'喜欢喝什么茶' in text})
        time.sleep(.3)
        observed={name:cache_evidence(name,offsets[name]) for name in limits}
        checks={name+'_logged_bound':bool(observed[name]['states']) and all(s['limit_mib']==limits[name] and s['mib']<=limits[name]+.001 for s in observed[name]['states']) for name in limits}
        checks.update({name+'_eviction_observed':observed[name]['evictions']>0 for name in limits})
        after=services.state(['llama','asr','speech'],health=True)
        checks.update(text_requests_echoed=all(r['tag_present'] for r in text_rows),audio_questions_preserved=all(r['question_preserved'] for r in audio_rows),
            services_ready=after['backend_status']=='READY',no_worker_restart=all(before['components'][name]['main_pid']==after['components'][name]['main_pid'] for name in ('llama','asr','speech')))
        report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'commands':commands,
            'cache':observed,'text_cases':text_rows,'audio_cases':audio_rows,'input_sha256':sha256(audio),
            'resources':monitor.report(),'backends_before':before,'backends_after':after,
            'scope':'20 unique long synthetic text requests plus20 variants of one synthetic spoken question(padded silence),real nativeLLM/ASR and OVS route. Native logs show host-cache size/eviction;not a whole-process memory cap,independent quality,latency,physical audio or8h acceptance. Data-restore fault workload may overlap;no timing claims.'}
        write_json(args.output,report);print(json.dumps({'status':report['status'],'checks':checks,'cache':{name:{k:v for k,v in values.items() if k!='states'} for name,values in observed.items()}},ensure_ascii=False),flush=True)
        return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
