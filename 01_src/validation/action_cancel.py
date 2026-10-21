#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cancel real model turns immediately after a durable action receipt arrives."""
import argparse
import asyncio
import json
from pathlib import Path
import uuid
import httpx
from websockets.asyncio.client import connect


async def probe(args):
    report={'scope':'Actual model/SQL receipts followed by Web cancellation; file output, not physical audio','cases':[]}
    async with httpx.AsyncClient(base_url=args.base_url,timeout=5,trust_env=False) as client:
        async def state():return (await client.get('/api/state')).raise_for_status().json()
        async def wait_turn(key):
            async with asyncio.timeout(90):
                while True:
                    current=await state();last=current['last_result']
                    if current['active_turn'] is None and last and last['turn_id']==key:return last
                    await asyncio.sleep(.05)
        async with connect(args.base_url.replace('http://','ws://',1)+'/ws',max_size=2**21) as ws:
            try:
                for name,prompt in [('create_note','请创建笔记，标题为取消边界测试，内容为已提交内容应该保留。'),
                                    ('create_reminder','请在五分钟后提醒我检查部署记录。')]:
                    request=uuid.uuid4().hex
                    accepted=(await client.post('/api/chat',json={'text':prompt,'request_id':request})).raise_for_status().json()
                    async with asyncio.timeout(90):
                        while True:
                            event=json.loads(await ws.recv())
                            if event.get('turn_id')!=accepted['turn_id']:continue
                            if event['type']=='done':raise RuntimeError('Turn finished before a durable receipt was observed')
                            if event['type']=='action_receipt' and event['receipt'].get('committed'):
                                receipt=event['receipt'];assert receipt['name']==name;break
                    cancelled=(await client.post('/api/cancel',json={'turn_id':accepted['turn_id']})).raise_for_status().json()
                    assert cancelled['accepted']
                    result=await wait_turn(accepted['turn_id'])
                    assert result['status']=='cancelled' and result['resources_released'] and not result['text']
                    assert result['actions_committed'][0]['result']==receipt['result']
                    assert (await client.get('/api/audio/'+accepted['turn_id'])).status_code==404
                    replay=(await client.post('/api/chat',json={'text':prompt,'request_id':request})).raise_for_status().json()
                    repeated=await wait_turn(replay['turn_id'])
                    assert repeated['status']=='complete' and repeated['replayed'] and 'model_rounds' not in repeated
                    assert repeated['actions'][0]['result']==receipt['result']
                    report['cases'].append({'tool':name,'receipt':receipt,'cancelled_result':result,'replayed_result':repeated})
                    if name=='create_reminder':
                        key=receipt['result']['reminder_id']
                        row=next(r for r in (await state())['actions']['reminders'] if r['id']==key)
                        assert row['status']=='scheduled'
                        cleanup=(await client.post('/api/chat',json={'text':'请取消编号为'+key+'的提醒。'})).raise_for_status().json()
                        cleaned=await wait_turn(cleanup['turn_id'])
                        assert cleaned['status']=='complete'
                        assert next(r for r in (await state())['actions']['reminders'] if r['id']==key)['status']=='cancelled'
                report['status']='PASSED_WITH_DECLARED_SCOPE'
            except Exception as exc:
                report.update(status='FAILED',error=str(exc));raise
            finally:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'tools':[c['tool'] for c in report['cases']]},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8093');parser.add_argument('--output',type=Path,required=True)
    asyncio.run(probe(parser.parse_args()))


if __name__ == "__main__":
    main()
