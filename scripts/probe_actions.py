#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Own an isolated Web instance to verify real tools, retries and scheduler restart.

Use a dedicated profile/database and file audio output. This writes test notes
and reminders; it never opens the old project's business databases.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import httpx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-note-result',type=Path);parser.add_argument('--port',type=int,default=8093)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'created_at':datetime.now(timezone.utc).isoformat(),'scope':'Actual local LLM tools, SQLAlchemy/SQLite and APScheduler with file speech; not physical audio or B0'}
    process=None;log=args.output.with_suffix('.log').open('w');client=httpx.Client(base_url=f'http://127.0.0.1:{args.port}',timeout=5,trust_env=False)
    def state():return client.get('/api/state').raise_for_status().json()
    def wait(predicate,seconds=90):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            value=predicate()
            if value:return value
            time.sleep(.1)
        raise RuntimeError('Action probe observation timed out')
    def start(disabled=False):
        nonlocal process
        command=[sys.executable,'run_web_vision_chat.py','--config',str(args.config),'--no-camera','--host','127.0.0.1','--port',str(args.port)]
        if disabled:command.append('--no-tools')
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        def ready():
            if process.poll() is not None:raise RuntimeError('Application exited during startup')
            try:return state()['ready']
            except httpx.HTTPError:return False
        wait(ready,20)
    def stop():
        nonlocal process
        if process is None:return
        process.terminate()
        try:process.wait(20)
        except subprocess.TimeoutExpired:process.kill();process.wait();raise
        process=None
    def ask(text,key=None,expected='complete'):
        request_id=key or uuid.uuid4().hex
        accepted=client.post('/api/chat',json={'text':text,'request_id':request_id}).raise_for_status().json()
        def complete():
            current=state();last=current['last_result']
            return last if current['active_turn'] is None and last and last['turn_id']==accepted['turn_id'] else None
        result=wait(complete)
        assert result['status']==expected,result
        if expected=='complete':assert client.get(result['audio_url']).status_code==200
        return result
    def operation(result,name):
        return next(item['result'] for item in result['actions_committed'] if item['name']==name)
    def reminder(key):return next(r for r in state()['actions']['reminders'] if r['id']==key)
    try:
        start()
        if args.seed_note_result:
            report['note']=json.loads(args.seed_note_result.read_text())['last_result']
        else:report['note']=ask('帮我创建一条笔记，标题为部署检查，内容为先测基线再优化。')
        note=operation(report['note'],'create_note');before=state()['actions']['notes']
        report['replay']=ask(report['note']['prompt'],report['note']['request_id'])
        assert report['replay']['replayed'] and 'model_rounds' not in report['replay'] and state()['actions']['notes']==before
        report['conflict']=ask('与原请求不同的内容',report['note']['request_id'],expected='failed')
        assert state()['actions']['notes']==before
        report['create_reminder']=ask('请在三十秒后提醒我喝水。')
        created=operation(report['create_reminder'],'create_reminder');key=created['reminder_id']
        due=datetime.fromisoformat(created['due_at']);remaining=(due-datetime.now(timezone.utc)).total_seconds()
        assert 0<remaining<=35,remaining
        stop();print('Application stopped before due time; waiting for overdue restart',flush=True)
        time.sleep(remaining+.3)
        start()
        wait(lambda:reminder(key)['status']=='delivered' and state()['active_turn'] is None)
        report['recovered_notification']=state()['last_result'];report['reminder_after_delivery']=reminder(key)
        assert report['recovered_notification']['notification_id']==key
        assert report['recovered_notification']['speech']['receipt']['kind']=='file_written'
        assert client.get(report['recovered_notification']['audio_url']).status_code==200
        assert 'model_rounds' not in report['recovered_notification']
        stop();start();time.sleep(1.5)
        assert reminder(key)['status']=='delivered' and state()['last_result'] is None
        report['delivered_notice_not_repeated_after_restart']=True
        report['query_notes']=ask('查询最近的笔记。')
        assert any(n['note_id']==note['note_id'] for n in operation(report['query_notes'],'list_notes')['notes'])
        report['query_reminders']=ask('查询最近的提醒。')
        assert any(r['reminder_id']==key for r in operation(report['query_reminders'],'list_reminders')['reminders'])
        report['second_reminder']=ask('请在六十秒后提醒我站起来活动。')
        future=operation(report['second_reminder'],'create_reminder')['reminder_id']
        report['cancel_reminder']=ask('请取消编号为'+future+'的提醒。')
        assert operation(report['cancel_reminder'],'cancel_reminder')['reminder_id']==future
        assert reminder(future)['status']=='cancelled' and state()['actions']['notes']==before
        stop();start(disabled=True)
        assert not state()['actions']['enabled']
        report['disabled_reply']=ask('帮我新建一条持久笔记，内容为功能关闭测试。')
        assert not report['disabled_reply'].get('actions_committed')
        report['status']='PASSED_WITH_DECLARED_SCOPE'
    except Exception as exc:
        report.update(status='FAILED',error=str(exc));raise
    finally:
        stop();client.close();log.close()
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'note_id':note['note_id'],'reminder_id':key,
        'restart_delivery':report['reminder_after_delivery']['status'],'next_reminder_cancelled':future,
        'disabled_reply':report['disabled_reply']['text']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
