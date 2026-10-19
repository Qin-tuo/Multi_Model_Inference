# SPDX-License-Identifier: Apache-2.0
import threading
import time
from types import SimpleNamespace
from fastapi.testclient import TestClient
from app.config import Config
from app.turn_control import TurnController
from app.web import Broadcaster, create_app


def done(control):
    deadline=time.monotonic()+2
    while control.state()['active_turn'] and time.monotonic()<deadline:time.sleep(.01)
    assert control.state()['active_turn'] is None
    return control.state()['last_result']


class Model:
    def __init__(self):self.contexts=[];self.started=threading.Event();self.release=threading.Event()
    def generate_stream(self,prompt,**kwargs):
        self.contexts.append(kwargs['few_shot'])
        if prompt=='cancel this':self.started.set();self.release.wait(2)
        if prompt=='fail this':raise RuntimeError('provider failed')
        yield 'answer '+prompt,{}
        yield '',{'done':True}
    def cancel(self,event):self.release.set()
    def idle(self):return True


def test_context_is_bounded_and_failed_cancelled_turns_are_not_answers():
    cfg=Config.load();cfg.session.max_turns=2;model=Model()
    control=TurnController(cfg,model,SimpleNamespace(send=lambda *a,**kw:None));control.start()
    try:
        for prompt in ['one','two','three']:
            control.submit(prompt);assert done(control)['status']=='complete'
        assert [m['content'] for m in model.contexts[-1]]==['one','answer one','two','answer two']
        assert control.session_state()['retained_turns']==2
        key=control.submit('cancel this')['turn_id'];assert model.started.wait(1);control.cancel(key)
        assert done(control)['status']=='cancelled'
        control.submit('fail this');assert done(control)['status']=='failed'
        control.submit('last');assert done(control)['status']=='complete'
        assert [m['content'] for m in model.contexts[-1]]==['two','answer two','three','answer three']
    finally:model.release.set();control.close()


def test_clear_session_is_bound_to_current_id_and_rejects_while_busy():
    cfg=Config.load();model=Model();broadcaster=Broadcaster();control=TurnController(cfg,model,broadcaster);control.start()
    app=create_app(broadcaster,submit=control.submit,state=control.state,clear_history=control.clear_history)
    try:
        with TestClient(app) as client:
            old=client.get('/api/state').json()['session']['id']
            key=control.submit('cancel this')['turn_id'];assert model.started.wait(1)
            assert client.post('/api/session/clear',json={'session_id':old}).status_code==409
            control.cancel(key);done(control)
            control.submit('remember');done(control)
            cleared=client.post('/api/session/clear',json={'session_id':old})
            assert cleared.status_code==200 and cleared.json()['session']['id']!=old
            assert control.state()['last_result'] is None and control.session_state()['retained_turns']==0
            assert client.post('/api/session/clear',json={'session_id':old}).status_code==409
            control.submit('after clear');done(control);assert not model.contexts[-1]
    finally:model.release.set();control.close()


def test_restart_and_disabled_history_do_not_inherit_old_turns():
    cfg=Config.load();model=Model();control=TurnController(cfg,model,SimpleNamespace(send=lambda *a,**kw:None))
    control.start();control.submit('first');done(control);old=control.session_state()['id'];control.close()
    try:
        control.start();assert control.session_state()['id']!=old and control.state()['last_result'] is None
        control.submit('after restart');done(control);assert not model.contexts[-1]
        cfg.session.enabled=False
        control.submit('disabled');done(control);assert not model.contexts[-1]
    finally:control.close()


def test_visual_context_contains_only_delivered_text_and_no_image_bytes():
    control=TurnController(Config.load(),None,None)
    control._history.append({'user':'画面是什么','assistant':'一个杯子','visual':True})
    messages=control._context_messages()
    assert '原图已丢弃' in messages[0]['content']
    assert all(isinstance(m['content'],str) for m in messages)


def test_tts_failure_does_not_commit_a_complete_history_turn(tmp_path):
    class TTS:
        timeout=.1
        def synthesize(self,*args,**kwargs):return {'audio':None,'error':'synthesis failed'}
        def cancel(self,event):pass
        def idle(self):return True
    cfg=Config.load();cfg.audio.output_directory=str(tmp_path)
    control=TurnController(cfg,Model(),SimpleNamespace(send=lambda *a,**kw:None),tts=TTS());control.start()
    try:
        control.submit('reply fails to synthesize')
        assert done(control)['status']=='failed'
        assert control.session_state()['retained_turns']==0
        assert not list(tmp_path.glob('*.wav'))
    finally:control.close()
