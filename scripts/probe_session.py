#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify actual multi-turn context and clearing on an isolated empty Web app."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import httpx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8093')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report={'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Actual model and Web session; no persistent memory, physical voice or B0'}
    try:
        with httpx.Client(base_url=args.base_url,timeout=5,trust_env=False) as client:
            def state():return client.get('/api/state').raise_for_status().json()
            def ask(text):
                key=client.post('/api/chat',json={'text':text}).raise_for_status().json()['turn_id']
                end=time.monotonic()+90
                while time.monotonic()<end:
                    s=state()
                    if s['active_turn'] is None and s['last_result'] and s['last_result']['turn_id']==key:
                        result=s['last_result']
                        assert result['status']=='complete',result
                        assert result['metadata']['input_tokens_preflight']==result['metadata']['usage']['prompt_tokens']
                        return result
                    time.sleep(.05)
                raise RuntimeError('Session probe timed out')
            initial=state()
            if initial['active_turn'] or initial['session']['retained_turns']:
                raise RuntimeError('Run this probe on an isolated empty application')
            report['initial_session']=initial['session']
            report['first']=ask('本次会话的测试代号是青鹭63B。请只回复收到。')
            report['recall']=ask('本次会话的测试代号是什么？只回答代号；没有提供就回答不知道。')
            report['recall_scoring']='Ignore whitespace only; raw answer remains recorded'
            assert '青鹭63B' in ''.join(report['recall']['text'].split()),report['recall']['text']
            old=initial['session']['id']
            cleared=client.post('/api/session/clear',json={'session_id':old}).raise_for_status().json()
            assert cleared['session']['id']!=old and cleared['session']['retained_turns']==0
            assert client.post('/api/session/clear',json={'session_id':old}).status_code==409
            report['after_clear']=ask('本次会话的测试代号是什么？只回答代号；没有提供就回答不知道。')
            assert '青鹭63B' not in ''.join(report['after_clear']['text'].split()) and '不知道' in report['after_clear']['text']
            assert report['after_clear']['metadata']['history_pairs_kept']==0
            report['budget_turns']=[]
            filler='请保留以下测试背景，但只回复收到。'+'部署任务需要记录线程、队列和模型输入输出。'*65
            for _ in range(3):report['budget_turns'].append(ask(filler))
            assert any(r['metadata']['history_pairs_dropped']>0 for r in report['budget_turns'])
            report['final_session']=state()['session']
            assert report['final_session']['retained_turns']<=report['final_session']['max_turns']
            report['status']='PASSED_WITH_DECLARED_SCOPE'
    except Exception as exc:
        report.update(status='FAILED',error=str(exc));raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'recall':report['recall']['text'],'after_clear':report['after_clear']['text'],
                      'budget_trims':[r['metadata']['history_pairs_dropped'] for r in report['budget_turns']]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
