#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check cancellation UI in Firefox using a running local geckodriver."""
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
                end=time.monotonic()+25
                while time.monotonic()<end:
                    if js(script):return
                    time.sleep(.1)
                raise RuntimeError('Browser condition timed out: '+script)
            call('POST',f'/session/{session}/url',{'url':args.base_url})
            wait("return document.querySelector('#connStatus').textContent.includes('Connected') && !document.querySelector('#sendText').disabled")
            js("""
                window.priorAnswers=document.querySelectorAll('.msg.assistant').length;
                window.cancelControls=0;
                const originalSend=WebSocket.prototype.send;
                WebSocket.prototype.send=function(data){if(JSON.parse(data).type==='cancel') window.cancelControls++;return originalSend.call(this,data);};
                document.querySelector('#textInput').value='请详细介绍十种学习编程的方法，每条解释具体步骤。';
                document.querySelector('#sendText').click();
            """)
            wait("return document.querySelectorAll('.msg.assistant').length>window.priorAnswers && !!Array.from(document.querySelectorAll('.msg.assistant .content')).at(-1)?.textContent.length && !document.querySelector('#cancelTurn').disabled")
            js("document.querySelector('#cancelTurn').click()")
            wait("return document.querySelector('#pipeStatus').textContent==='等待输入'")
            time.sleep(1.3)
            report=js("""return {
                phase:document.querySelector('#pipeStatus').textContent,
                send_enabled:!document.querySelector('#sendText').disabled,
                cancel_disabled:document.querySelector('#cancelTurn').disabled,
                assistant_text:Array.from(document.querySelectorAll('.msg.assistant .content')).at(-1)?.textContent,
                meta:Array.from(document.querySelectorAll('.msg.assistant .meta')).at(-1)?.textContent,
                error:document.querySelector('#inputError').textContent};""")
            assert report['phase']=='等待输入' and report['send_enabled'] and report['cancel_disabled']
            assert report['assistant_text']=='' and '已取消' in report['meta'] and not report['error']
            assert js('return window.cancelControls')==1
            report.update(status='PASSED',browser='Firefox via WebDriver',cancel_transport='existing WebSocket',
                          scope='Real browser cancellation via existing WebSocket; no physical audio')
            screenshot=call('GET',f'/session/{session}/screenshot')
            args.output.with_suffix('.png').write_bytes(base64.b64decode(screenshot))
        except Exception as exc:
            report.update(status='FAILED',error=str(exc));raise
        finally:
            if session:call('DELETE',f'/session/{session}')
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
