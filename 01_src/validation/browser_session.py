#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check session clearing and a delayed clear reply in real Firefox."""
import argparse
import base64
import json
from pathlib import Path
import time
import httpx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8093')
    parser.add_argument('--webdriver',default='http://127.0.0.1:4445')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report={};session=None
    with httpx.Client(base_url=args.webdriver,timeout=30,trust_env=False) as client:
        def call(method,path,data=None):
            value=client.request(method,path,json=data).raise_for_status().json()['value']
            if isinstance(value,dict) and value.get('error'):raise RuntimeError(value)
            return value
        try:
            session=call('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'args':['-headless']}}}})['sessionId']
            def js(script):return call('POST',f'/session/{session}/execute/sync',{'script':script,'args':[]})
            def wait(script):
                end=time.monotonic()+30
                while time.monotonic()<end:
                    if js(script):return
                    time.sleep(.1)
                raise RuntimeError('Browser condition timed out: '+script)
            call('POST',f'/session/{session}/url',{'url':args.base_url})
            wait("return !document.querySelector('#clearSession').disabled")
            js("document.querySelector('#textInput').value='请只回复：旧会话完成。';document.querySelector('#sendText').click();")
            wait("return document.querySelector('#sessionInfo').textContent.includes('保留 1 轮') && !document.querySelector('#clearSession').disabled")
            js("""window.originalFetch=window.fetch;
                window.fetch=async (...args)=>{const r=await window.originalFetch(...args);
                if(args[0]==='/api/session/clear') {window.clearReplyPending=true;await new Promise(resolve=>setTimeout(resolve,2000));window.clearReplyPending=false;}
                return r;}; document.querySelector('#clearSession').click();""")
            wait("return document.querySelectorAll('.msg').length===0 && !document.querySelector('#sendText').disabled")
            report['clear_http_reply_still_pending']=js('return window.clearReplyPending===true')
            assert report['clear_http_reply_still_pending']
            js("document.querySelector('#textInput').value='请只回复：新会话开始。';document.querySelector('#sendText').click();")
            wait("return document.querySelector('.msg.assistant .content')?.textContent.includes('新会话开始') && document.querySelector('#pipeStatus').textContent==='等待输入'")
            wait('return window.clearReplyPending===false')
            report.update(js("""return {messages:document.querySelectorAll('.msg').length,
                text:document.querySelector('#messages').textContent,session_label:document.querySelector('#sessionInfo').textContent,
                error:document.querySelector('#inputError').textContent};"""))
            assert report['messages']==2 and '新会话开始' in report['text'] and '旧会话完成' not in report['text']
            assert not report['error'] and '保留 1 轮' in report['session_label']
            report.update(status='PASSED',scope='Actual Firefox session clear and delayed HTTP response; local model text output')
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.with_suffix('.png').write_bytes(base64.b64decode(call('GET',f'/session/{session}/screenshot')))
        except Exception as exc:
            report.update(status='FAILED',error=str(exc));raise
        finally:
            if session:call('DELETE',f'/session/{session}')
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
