#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record live gesture observations; optional explicit palm-triggered stop test.

Start the normal camera application first. No camera images are saved. Raw
landmarks stay in the requested local output; publish only reviewed aggregates.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import httpx
from websockets.asyncio.client import connect


async def probe(args):
    if not 1<=args.seconds<=300:raise ValueError('Duration must be 1–300 seconds')
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'scope':'Live perception/event observation; not annotated gesture accuracy or physical audio',
            'samples':[],'actions':[],'results':{},'prime_stop_test_requested':args.prime_stop_test}
    seen=set();known_turns=set();primed=None
    async with httpx.AsyncClient(base_url=args.base_url,timeout=5,trust_env=False) as client:
        async with connect(args.base_url.replace('http://','ws://',1)+'/ws',max_size=2**21) as ws:
            async def receive():
                async for raw in ws:
                    event=json.loads(raw)
                    if event.get('type')=='gesture':
                        report['actions'].append(event)
                        if event.get('turn_id'):known_turns.add(event['turn_id'])
                        print(json.dumps({'gesture':event['event']['kind'],'accepted':event['accepted'],'reason':event['reason']},ensure_ascii=False),flush=True)
            receiver=asyncio.create_task(receive())
            try:
                deadline=time.monotonic()+args.seconds
                while time.monotonic()<deadline:
                    if receiver.done():receiver.result();raise RuntimeError('WebSocket ended during observation')
                    state=(await client.get('/api/state')).raise_for_status().json()
                    scene=state['perception'];gestures=state['gestures'];frame=scene.get('frame')
                    if frame:
                        key=(frame['source_epoch'],frame['frame_id'])
                        if key not in seen:
                            seen.add(key);report['samples'].append({'perception':scene,'gestures':gestures,
                                'active_turn':state['active_turn'],'phase':state['phase']})
                    if args.prime_stop_test and primed is None and state['active_turn'] is None:
                        if any(t.get('open_palm') and not t.get('latched') and t.get('stationary_seconds',0)<.5 for t in gestures.get('tracks',[])):
                            response=(await client.post('/api/chat',json={'text':'请从一数到一百，每个数字独立写一行，不要省略。'})).raise_for_status().json()
                            primed=response['turn_id'];known_turns.add(primed);report['primed_turn']=primed
                            print('Stop-test request started on newly raised palm',flush=True)
                    result=state.get('last_result')
                    if result and result['turn_id'] in known_turns and result.get('resources_released'):
                        report['results'][result['turn_id']]=result
                    await asyncio.sleep(.1)
                report['status']='OBSERVED'
            except Exception as exc:
                report.update(status='FAILED',error=str(exc));raise
            finally:
                receiver.cancel();await asyncio.gather(receiver,return_exceptions=True)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'distinct_frames':len(seen),'events':len(report['actions']),
        'hand_positive_frames':sum(bool(s['perception']['hands']) for s in report['samples']),
        'completed_results':len(report['results'])},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8090')
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prime-stop-test',action='store_true',help='Submit a counting request when a fresh open palm first appears, to test cancellation while it remains held')
    asyncio.run(probe(parser.parse_args()))
