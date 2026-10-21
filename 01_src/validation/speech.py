#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the running application with a bounded PCM WAV and retrieve its reply."""
import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse
import wave
import httpx
from websockets.asyncio.client import connect


async def probe(args):
    endpoint=urlparse(args.base_url)
    if endpoint.scheme!='http' or endpoint.hostname not in {'127.0.0.1','localhost','::1'}:
        raise ValueError('Probe requires a local HTTP endpoint')
    if args.input.stat().st_size>964096:raise ValueError('Input exceeds the maximum 30s PCM WAV limit')
    data=args.input.read_bytes()
    async with httpx.AsyncClient(base_url=args.base_url,timeout=10,trust_env=False) as client:
        async with connect(args.base_url.replace('http://','ws://',1)+'/ws',max_size=2**21) as ws:
            accepted=(await client.post('/api/audio',content=data,headers={'Content-Type':'audio/wav'})).raise_for_status().json()
            async with asyncio.timeout(180):
                while True:
                    event=json.loads(await ws.recv())
                    if event.get('turn_id')==accepted['turn_id'] and event['type']=='done':break
            state=(await client.get('/api/state')).raise_for_status().json()
            result=state['last_result']
            if result['turn_id']!=accepted['turn_id']:raise RuntimeError('Turn ownership changed during probe')
            if result['status']!='complete':raise RuntimeError(result.get('error','Speech turn failed'))
            if 'audio_url' not in result:raise RuntimeError('This probe requires file output mode')
            reply=(await client.get(result['audio_url'])).raise_for_status().content
            with wave.open(io.BytesIO(reply),'rb') as wav:
                fmt={'rate':wav.getframerate(),'channels':wav.getnchannels(),'width':wav.getsampwidth(),'frames':wav.getnframes()}
            if (fmt['rate'],fmt['channels'],fmt['width'])!=(16000,1,2):raise ValueError('Unexpected reply WAV format')
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(reply)
            report={'scope':'Real file/model I/O, not physical audio or a performance baseline','input_sha256':hashlib.sha256(data).hexdigest(),
                    'output_sha256':hashlib.sha256(reply).hexdigest(),'output_format':fmt,'result':result}
            args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8090')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(probe(parser.parse_args()))


if __name__=='__main__':main()
