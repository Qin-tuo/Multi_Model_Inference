# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from datetime import datetime,timedelta,timezone
import json
import threading
import time
from types import SimpleNamespace
import uuid
import pytest
from sqlalchemy import select,func
from sqlalchemy.engine import Connection
from sqlalchemy.sql.dml import Insert
from app.actions.store import ActionStore,NOTES,REMINDERS,OPERATIONS
from app.config import Config,ActionsConfig
from app.turn_control import TurnController


def wait(predicate,seconds=4):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.02)
    assert predicate(),'Condition did not become true'


@pytest.fixture
def store(tmp_path):
    action=ActionStore(replace(ActionsConfig(),database=str(tmp_path/'actions.sqlite')))
    try:yield action
    finally:action.close()


def request(store):
    key=uuid.uuid4().hex;store.begin_request(key,'input-digest');return key


def test_atomic_note_receipt_idempotence_conflict_and_restart(store):
    key=request(store);args={'title':'检查','body':'先测基线'}
    first=store.execute(key,0,'create_note',args)
    assert store.execute(key,0,'create_note',args)==first
    assert store.execute(key,1,'create_note',args)['replayed']
    with pytest.raises(ValueError):store.execute(key,0,'create_note',{**args,'body':'changed'})
    with pytest.raises(ValueError):store.begin_request(key,'different-input')
    store.finish_request(key,'cancelled');store.close()
    reopened=ActionStore(store.config)
    try:
        replay=reopened.begin_request(key,'input-digest')
        assert replay['replay'] and replay['status']=='cancelled' and replay['receipts'][0]['result']==first
        assert reopened.snapshot()['notes']==1
    finally:reopened.close()


def test_operation_receipt_failure_rolls_back_the_note(store,monkeypatch):
    key=request(store);original=Connection.execute
    def fail_receipt(self,statement,*args,**kwargs):
        if isinstance(statement,Insert) and statement.table.name==OPERATIONS.name:raise RuntimeError('simulated receipt write failure')
        return original(self,statement,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Connection,'execute',fail_receipt)
        with pytest.raises(RuntimeError,match='receipt write'):store.execute(key,0,'create_note',{'title':'test','body':'body'})
    assert store.snapshot()['notes']==0 and store.receipts(key)==[]


def test_one_owner_lock_does_not_disturb_existing_owner(store):
    with pytest.raises(RuntimeError,match='owns this action database'):ActionStore(store.config)
    assert store.begin_request(uuid.uuid4().hex,'test')['replay'] is False


def test_scheduler_failure_after_commit_is_visible_and_recoverable(store,monkeypatch):
    key=request(store);due=(datetime.now(timezone.utc)+timedelta(seconds=.5)).isoformat()
    with monkeypatch.context() as patch:
        def unavailable(*args,**kwargs):raise RuntimeError('scheduler unavailable')
        patch.setattr(store.scheduler,'add_job',unavailable)
        result=store.execute(key,0,'create_reminder',{'text':'保留提醒','due_at':due})
        assert result['scheduling_confirmed'] is False and store.snapshot()['reminders'][0]['status']=='scheduled'
        assert store.receipts(key)[0]['result']['reminder_id']==result['reminder_id']
    store.reconcile()
    wait(lambda:store.snapshot()['reminders'][0]['status']=='due')


def test_real_scheduler_survives_restart_and_unknown_delivery_does_not_repeat(store):
    key=request(store);due=(datetime.now(timezone.utc)+timedelta(seconds=.5)).isoformat()
    created=store.execute(key,0,'create_reminder',{'text':'喝水','due_at':due});store.finish_request(key,'complete')
    store.close();time.sleep(.6)
    reopened=ActionStore(store.config)
    try:
        wait(lambda:reopened.snapshot()['reminders'][0]['status']=='due')
        notice=reopened.claim_due();assert notice['id']==created['reminder_id']
        reopened.begin_delivery(notice,'file');reopened.close()
        final=ActionStore(store.config)
        try:
            row=final.snapshot()['reminders'][0]
            assert row['status']=='uncertain' and final.claim_due() is None
        finally:final.close()
    finally:reopened.close()


def test_reminder_cancelled_before_delivery_cannot_be_output(store):
    key=request(store);due=(datetime.now(timezone.utc)+timedelta(seconds=.4)).isoformat()
    value=store.execute(key,0,'create_reminder',{'text':'test','due_at':due})
    wait(lambda:store.snapshot()['reminders'][0]['status']=='due')
    notice=store.claim_due()
    store.execute(key,1,'cancel_reminder',{'reminder_id':value['reminder_id']})
    with pytest.raises(RuntimeError):store.begin_delivery(notice,'file')
    assert store.snapshot()['reminders'][0]['status']=='cancelled'


def test_bad_arguments_unknown_tools_and_past_time_do_not_write(store):
    key=request(store)
    for name,args in [('shell',{}),('create_note',{'title':'x','body':'y','path':'bad'}),
        ('create_reminder',{'text':'x','delay_seconds':True}),
        ('create_reminder',{'text':'x','due_at':'2000-01-01T00:00:00+00:00'}),
        ('create_reminder',{'text':'x','due_at':'2099-01-01T00:00:00','delay_seconds':2})]:
        with pytest.raises(ValueError):store.execute(key,0,name,args)
    assert store.snapshot()['notes']==0 and not store.snapshot()['reminders']


class Model:
    def __init__(self):self.calls=0;self.started=threading.Event();self.release=threading.Event();self.wait_before_call=False
    def idle(self):return True
    def cancel(self,event):self.release.set()
    def generate_stream(self,*args,**kwargs):
        self.calls+=1;self.started.set()
        if self.wait_before_call:self.release.wait(2)
        if not kwargs.get('tool_messages'):
            yield '',{'done':True,'tool_calls':[
                {'id':'call-note','type':'function','function':{'name':'create_note','arguments':json.dumps({'title':'记录','body':'保留已提交内容'})}},
                {'id':'call-reminder','type':'function','function':{'name':'create_reminder','arguments':json.dumps({'text':'未执行提醒','delay_seconds':60})}}]}
        else:yield '操作已完成',{'done':True}


def test_cancel_after_first_commit_blocks_unstarted_second_tool_and_replay(store,monkeypatch):
    cfg=Config.load();model=Model();control=TurnController(cfg,model,SimpleNamespace(send=lambda *a,**kw:None),actions=store)
    original=store.execute
    def execute(*args):
        value=original(*args);control.cancel(reason='test_after_commit');return value
    monkeypatch.setattr(store,'execute',execute);control.start()
    try:
        key=uuid.uuid4().hex;control.submit('做两件事',request_id=key)
        wait(lambda:control.state()['active_turn'] is None)
        result=control.state()['last_result']
        assert result['status']=='cancelled' and len(result['actions_committed'])==1
        assert store.snapshot()['notes']==1 and not store.snapshot()['reminders']
        control.submit('做两件事',request_id=key);wait(lambda:control.state()['active_turn'] is None)
        assert control.state()['last_result']['replayed'] and model.calls==1 and store.snapshot()['notes']==1
    finally:control.close()


def test_cancel_before_model_call_dispatch_executes_no_tool(store):
    model=Model();model.wait_before_call=True
    control=TurnController(Config.load(),model,SimpleNamespace(send=lambda *a,**kw:None),actions=store);control.start()
    try:
        key=control.submit('稍后创建笔记')['turn_id'];assert model.started.wait(1);control.cancel(key)
        wait(lambda:control.state()['active_turn'] is None)
        assert control.state()['last_result']['status']=='cancelled'
        assert store.snapshot()['notes']==0 and not store.snapshot()['reminders']
    finally:model.release.set();control.close()


def test_due_notice_uses_same_output_and_can_finish_without_model(store,tmp_path):
    import numpy as np
    class TTS:
        timeout=.1
        def synthesize(self,*a,**kw):return {'audio':np.ones(160,dtype=np.int16),'sample_rate':16000}
        def idle(self):return True
        def cancel(self,event):pass
    key=request(store);store.execute(key,0,'create_reminder',{'text':'提醒内容','delay_seconds':1})
    cfg=Config.load();cfg.audio.output_directory=str(tmp_path/'audio');model=Model()
    control=TurnController(cfg,model,SimpleNamespace(send=lambda *a,**kw:None,ptt_active=True),actions=store,tts=TTS());control.start()
    try:
        wait(lambda:store.snapshot()['reminders'][0]['status']=='delivered')
        wait(lambda:control.state()['active_turn'] is None)
        assert model.calls==0 and control.state()['last_result']['notification_id']
        assert control.state()['last_result']['speech']['receipt']['kind']=='file_written'
        assert control.session_state()['retained_turns']==0
    finally:control.close()


def test_stopping_notice_keeps_note_and_reminder_and_does_not_auto_repeat(store,tmp_path):
    from app.request_io import TurnCancelled
    import numpy as np
    class TTS:
        timeout=.1
        started=threading.Event()
        def synthesize(self,text,cancel_event=None):
            if text.startswith('提醒：'):
                self.started.set();assert cancel_event.wait(3);raise TurnCancelled('stopped')
            return {'audio':np.ones(160,dtype=np.int16),'sample_rate':16000}
        def idle(self):return True
        def cancel(self,event):pass
    class Chat:
        def generate_stream(self,*a,**kw):yield '你好',{'done':True}
        def idle(self):return True
        def cancel(self,event):pass
    key=request(store);store.execute(key,0,'create_note',{'title':'留下','body':'已提交笔记'})
    reminder=store.execute(key,1,'create_reminder',{'text':'待停止','delay_seconds':1})
    cfg=Config.load();cfg.audio.output_directory=str(tmp_path/'audio');tts=TTS()
    control=TurnController(cfg,Chat(),SimpleNamespace(send=lambda *a,**kw:None,ptt_active=True),actions=store,tts=tts);control.start()
    try:
        assert tts.started.wait(3);turn=control.state()['active_turn'];control.cancel(turn)
        wait(lambda:control.state()['active_turn'] is None)
        row=store.snapshot()['reminders'][0]
        assert row['id']==reminder['reminder_id'] and row['status']=='stopped'
        assert store.snapshot()['notes']==1 and control.audio_file(turn) is None
        control.submit('你好');wait(lambda:control.state()['active_turn'] is None)
        assert control.state()['last_result']['status']=='complete' and store.claim_due() is None
    finally:control.close()


def test_capacity_rejects_new_writes_without_deleting_saved_notes(store):
    store.config.max_records=1
    first=request(store);store.execute(first,0,'create_note',{'title':'原有','body':'保留'})
    second=request(store)
    with pytest.raises(RuntimeError,match='capacity'):store.execute(second,0,'create_note',{'title':'新增','body':'不能覆盖原有'})
    assert store.snapshot()['notes']==1 and store.receipts(second)==[]


def test_cancel_cannot_relabel_an_already_delivered_reminder(store):
    key=request(store);item=store.execute(key,0,'create_reminder',{'text':'already done','due_at':(datetime.now(timezone.utc)+timedelta(seconds=.3)).isoformat()})
    wait(lambda:store.snapshot()['reminders'][0]['status']=='due')
    notice=store.claim_due();store.begin_delivery(notice,'file');receipt={'kind':'file_written','path':'owned-output.wav'}
    store.finish_delivery(notice,receipt)
    result=store.execute(key,1,'cancel_reminder',{'reminder_id':item['reminder_id']})
    row=store.snapshot()['reminders'][0]
    assert result['status']=='already_delivered' and row['status']=='delivered' and json.loads(row['receipt'])==receipt


def test_saved_image_request_replay_does_not_require_a_new_camera_frame(store):
    class OneNote(Model):
        def generate_stream(self,*args,**kwargs):
            self.calls+=1
            if not kwargs.get('tool_messages'):
                yield '',{'done':True,'tool_calls':[{'id':'note','type':'function','function':{'name':'create_note','arguments':'{"title":"原图","body":"原图记录"}'}}]}
            else:yield '已保存',{'done':True}
    frame=SimpleNamespace(jpeg=b'original',metadata=lambda:{'frame_id':1})
    camera=SimpleNamespace(source_epoch='camera',health_check=lambda:True,snapshot=lambda:frame,_base64_jpeg=lambda raw:'b3JpZ2luYWw=')
    model=OneNote();control=TurnController(Config.load(),model,SimpleNamespace(send=lambda *a,**kw:None),camera,actions=store);control.start()
    try:
        key=uuid.uuid4().hex;control.submit('根据原图保存笔记',True,request_id=key);wait(lambda:control.state()['active_turn'] is None)
        assert control.state()['last_result']['status']=='complete' and store.snapshot()['notes']==1
        camera.health_check=lambda:False;camera.snapshot=lambda:None
        control.submit('根据原图保存笔记',True,request_id=key);wait(lambda:control.state()['active_turn'] is None)
        result=control.state()['last_result']
        assert result['replayed'] and result['mode']=='receipt_replay' and 'frame' not in result
        assert model.calls==2 and store.snapshot()['notes']==1
    finally:control.close()
