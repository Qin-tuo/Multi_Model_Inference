#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record a bounded set of real current-frame perception outputs from the app."""
import argparse
import json
from pathlib import Path
import time
import httpx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8090')
    parser.add_argument('--seconds',type=float,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.seconds<=60:raise ValueError('Observation duration must be 0–60 seconds')
    seen=set();samples=[];failures=[]
    with httpx.Client(timeout=5,trust_env=False) as client:
        deadline=time.monotonic()+args.seconds
        while time.monotonic()<deadline:
            response=client.get(args.base_url+'/api/state');response.raise_for_status()
            state=response.json()['perception']
            if state['ready']:
                key=(state['frame']['source_epoch'],state['frame']['frame_id'])
                if key not in seen:seen.add(key);samples.append(state)
            else:failures.append(state.get('error'))
            time.sleep(.2)
    result={'scope':'Actual published frames; counts are observations, not annotated accuracy or performance baseline',
            'samples':samples,'unready_observations':failures,'distinct_frames':len(samples),
            'person_positive_frames':sum(bool(x['persons']) for x in samples),
            'face_positive_frames':sum(bool(x['face']['face_detected']) for x in samples),
            'hand_positive_frames':sum(bool(x['hands']) for x in samples)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=False,indent=2))
    if not samples:raise RuntimeError('No valid perception frames observed')


if __name__=='__main__':main()
