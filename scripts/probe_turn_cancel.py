#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise real Web turn cancellation during ASR, generation and synthesis."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import httpx
from websockets.asyncio.client import connect


async def probe(args):
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'scope':'Real local Web and models; file I/O, no camera/physical audio/gestures or B0', 'cases':[]}
    async with httpx.AsyncClient(timeout=3,trust_env=False) as client:
        async def state():return (await client.get(args.base_url+'/api/state')).raise_for_status().json()
        async def wait(predicate,seconds=30):
            async with asyncio.timeout(seconds):
                while not await predicate():await asyncio.sleep(.02)
        async def free():return (await state())['active_turn'] is None
        async with connect(args.base_url.replace('http://','ws://',1)+'/ws') as ws:
            events=[]
            async def receive():
                async for message in ws:
                    event=json.loads(message)
                    if event.get('turn_id'):events.append(event)
            receiver=asyncio.create_task(receive())
            try:
                for kind in ['asr','llm','tts']:
                    await wait(free)
                    config=(await state())['effective_config']
                    async def busy():
                        if kind=='tts':
                            value=(await client.get(config['tts']['base_url']+'/capabilities')).raise_for_status().json()
                            return value['tts']['concurrency']['active']>0
                        url=config['stt']['native_base_url'] if kind=='asr' else config['llm']['base_url']
                        return any(slot['is_processing'] for slot in (await client.get(url+'/slots')).raise_for_status().json())
                    if await busy():raise RuntimeError('Backend already in use before probe')
                    if kind=='asr':
                        response=await client.post(args.base_url+'/api/audio',content=args.input.read_bytes(),headers={'Content-Type':'audio/wav'})
                    else:
                        response=await client.post(args.base_url+'/api/chat',json={'text':'请详细介绍十种学习编程的方法，每条解释具体步骤。'})
                    key=response.raise_for_status().json()['turn_id']
                    await wait(busy,10)
                    started=time.monotonic()
                    cancelled=(await client.post(args.base_url+'/api/cancel',json={'turn_id':key})).raise_for_status().json()
                    assert cancelled['accepted']
                    await wait(free)
                    result=(await state())['last_result']
                    assert result['turn_id']==key and result['status']=='cancelled' and not result['text']
                    assert result['resources_released'] and 'audio_url' not in result
                    assert (await client.get(args.base_url+'/api/audio/'+key)).status_code==404
                    async def saw_idle():return any(e.get('turn_id')==key and e.get('stage')=='idle' for e in events)
                    await wait(saw_idle,3)
                    owned=[event for event in events if event['turn_id']==key]
                    cancellation_index=next(i for i,e in enumerate(owned) if e['type']=='cancelled')
                    assert not any(e['type'] in ['token','transcript'] for e in owned[cancellation_index+1:])
                    case={'cancel_during':kind,'cancel_to_released_seconds':time.monotonic()-started,
                          'result':result,'late_output_after_cancel':False,'cancelled_audio_404':True}
                    # Verify an old UI request cannot cancel the next turn.
                    new=(await client.post(args.base_url+'/api/chat',json={'text':'只回答：恢复成功。'})).raise_for_status().json()['turn_id']
                    stale=(await client.post(args.base_url+'/api/cancel',json={'turn_id':key})).raise_for_status().json()
                    assert not stale['accepted']
                    await wait(free,90)
                    recovered=(await state())['last_result']
                    assert recovered['turn_id']==new and recovered['status']=='complete' and recovered['resources_released']
                    assert (await client.get(args.base_url+recovered['audio_url'])).status_code==200
                    case['next_turn_complete_with_audio']=True
                    report['cases'].append(case)
                report['status']='PASSED_WITH_DECLARED_SCOPE'
            finally:
                receiver.cancel();await asyncio.gather(receiver,return_exceptions=True)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':[{k:v for k,v in c.items() if k!='result'} for c in report['cases']]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8093')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(probe(parser.parse_args()))
