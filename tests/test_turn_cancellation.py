# SPDX-License-Identifier: Apache-2.0
"""Cancellation races using real sockets and owned subprocesses, without devices."""
import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.config import Config
from app.pipeline import SpeechOutput, Playback, play_audio
from app.request_io import RequestIO, TurnCancelled
from app.turn_control import TurnController
from app.web import Broadcaster, create_app


def until(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.01)
    assert predicate(), 'Condition did not become true'


@contextmanager
def stalled_http(headers):
    entered = threading.Event(); finish = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if headers:
                self.send_response(200); self.send_header('Content-Length', '1000000'); self.end_headers()
                self.wfile.write(b'x'); self.wfile.flush()
            entered.set(); finish.wait(5)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try: yield f'http://127.0.0.1:{server.server_port}', entered
    finally:
        finish.set(); server.shutdown(); server.server_close(); thread.join(2)


@pytest.mark.parametrize('headers',[False,True])
def test_cancel_interrupts_real_socket_before_headers_and_during_body(headers):
    io = RequestIO(); event = threading.Event(); errors = []
    with stalled_http(headers) as (url,entered):
        def read():
            try:
                with io.operation(event) as op, httpx.Client(timeout=4,trust_env=False) as client:
                    client.get(url, extensions=op.extensions)
            except Exception as exc: errors.append(exc)
        thread = threading.Thread(target=read); thread.start()
        assert entered.wait(2)
        started=time.monotonic(); event.set(); io.cancel(event); thread.join(1)
        assert not thread.is_alive() and time.monotonic()-started < 1
        assert len(errors)==1 and isinstance(errors[0],TurnCancelled) and io.idle()


def test_stale_cancel_cannot_abort_next_request_and_pre_cancel_never_starts():
    io=RequestIO(); old=threading.Event(); old.set()
    with pytest.raises(TurnCancelled):
        with io.operation(old): pytest.fail('Pre-cancelled operation entered')
    with io.operation(threading.Event()) as current:
        called=[]; current.abort=lambda:called.append(True)
        io.cancel(old)
        assert not called
        with pytest.raises(RuntimeError,match='Overlapping'):
            with io.operation(): pass


class Model:
    def __init__(self):
        self.started=threading.Event(); self.unblock=threading.Event(); self.released=threading.Event()
    def generate_stream(self,*args,**kwargs):
        self.started.set(); self.unblock.wait(3)
        yield 'late text',{}
        yield '',{'done':True}
    def cancel(self,event): self.unblock.set()
    def idle(self): return self.released.is_set()


def test_cancel_rejects_late_output_and_gates_new_turn_until_native_release():
    model=Model(); events=[]
    sink=SimpleNamespace(send=lambda event,**kwargs:events.append(event))
    control=TurnController(Config.load(),model,sink);control.start()
    try:
        key=control.submit('first')['turn_id'];assert model.started.wait(1)
        assert control.cancel(key)['accepted']
        until(lambda:control.state()['last_result'] is not None)
        assert control.state()['last_result']['status']=='cancelled'
        assert control.state()['last_result']['text']==''
        assert not any(e['type']=='token' for e in events)
        with pytest.raises(RuntimeError):control.submit('too early')
        model.released.set();until(lambda:control.state()['active_turn'] is None)
        second=control.submit('second')['turn_id']
        assert not control.cancel(key)['accepted']
        until(lambda:control.state()['active_turn'] is None)
        assert control.state()['last_result']['turn_id']==second
        assert control.state()['last_result']['status']=='complete'
    finally:model.released.set();control.close()


def test_unconfirmed_release_stays_visible_and_recovers():
    model=Model();model.unblock.set()
    control=TurnController(Config.load(),model,SimpleNamespace(send=lambda e,**kw:None));control.start()
    try:
        control.submit('test')
        until(lambda:control.state()['phase']=='release_unconfirmed',6)
        assert not control.state()['ready'] and not control.state()['release']['confirmed']
        with pytest.raises(RuntimeError):control.submit('blocked')
        model.released.set();until(lambda:control.state()['active_turn'] is None)
        assert control.state()['ready']
    finally:model.released.set();control.close()


class TTS:
    timeout=.1
    def synthesize(self,text,**kwargs):
        return {'audio':np.ones(160,dtype=np.int16),'sample_rate':16000}
    def cancel(self,event): pass


def test_cancel_between_staging_and_commit_keeps_completed_files(tmp_path):
    cfg=Config.load().audio;cfg.output_directory=str(tmp_path)
    completed=SpeechOutput(TTS(),cfg,'a'*32);completed.submit('已完成');receipt=completed.close()
    event=threading.Event()
    output=SpeechOutput(TTS(),cfg,'b'*32,cancel_event=event,defer_commit=True)
    output.submit('待提交');output.close()
    assert list(tmp_path.glob('*.partial')) and not (tmp_path/('reply-'+'b'*32+'.wav')).exists()
    event.set();output.cancel()
    with pytest.raises(TurnCancelled):output.publish()
    output.close(False)
    assert Path(receipt['path']).exists() and not list(tmp_path.glob('*.partial'))


def test_full_speech_queue_unblocks_on_cancel_and_does_not_start_pending_text(tmp_path):
    class SlowTTS(TTS):
        def __init__(self):self.started=threading.Event();self.calls=[]
        def synthesize(self,text,cancel_event=None):
            self.calls.append(text);self.started.set();assert cancel_event.wait(2)
            raise TurnCancelled('cancelled')
    tts=SlowTTS();cfg=Config.load().audio;cfg.output_directory=str(tmp_path)
    output=SpeechOutput(tts,cfg,'c'*32)
    output.submit('one');assert tts.started.wait(1)
    output.submit('two');output.submit('three')
    errors=[]
    def submit():
        try:output.submit('four')
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=submit);thread.start()
    output.cancel();thread.join(1);output.close(False)
    assert not thread.is_alive() and errors and tts.calls==['one']
    assert not list(tmp_path.iterdir())


def test_cancel_terminates_only_owned_playback_even_if_it_ignores_term(monkeypatch):
    actual=subprocess.Popen; started=threading.Event(); owned=[]
    # A stand-in player owns a real process and pipes, with no speaker attached.
    def launch(*args,**kwargs):
        process=actual([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(10)'],
                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        assert process.stdout.readline()==b'ready\n'
        owned.append(process);started.set();return process
    monkeypatch.setattr('app.pipeline.subprocess.Popen',launch)
    event=threading.Event();playback=Playback(event);errors=[]
    def play():
        try:play_audio(np.ones(160,dtype=np.int16),16000,'test-sink',playback)
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=play);thread.start();assert started.wait(1)
    event.set();playback.cancel();thread.join(2)
    assert not thread.is_alive() and owned[0].poll() is not None
    assert len(errors)==1 and isinstance(errors[0],TurnCancelled)


def test_cancelled_events_are_filtered_even_if_already_queued():
    event=threading.Event();q=asyncio.Queue(maxsize=2)
    Broadcaster._enqueue_latest(q,{'type':'token','text':'stale','_cancel_event':event})
    event.set()
    assert Broadcaster.visible(q.get_nowait()) is None
    Broadcaster._enqueue_latest(q,{'type':'token','_cancel_event':event})
    assert q.empty()
    assert Broadcaster.visible({'type':'done','status':'cancelled'})['status']=='cancelled'


def test_web_cancel_requires_specific_turn_and_returns_before_release():
    model=Model();broadcaster=Broadcaster();control=TurnController(Config.load(),model,broadcaster);control.start()
    try:
        with TestClient(create_app(broadcaster,submit=control.submit,cancel=control.cancel,state=control.state)) as client:
            assert client.post('/api/cancel',json={}).status_code==422
            key=client.post('/api/chat',json={'text':'test'}).json()['turn_id']
            assert model.started.wait(1)
            response=client.post('/api/cancel',json={'turn_id':key})
            assert response.status_code==202 and response.json()['accepted']
            assert not response.json()['resources_released']
            assert client.post('/api/chat',json={'text':'too early'}).status_code==409
            model.released.set();until(lambda:control.state()['active_turn'] is None)
    finally:model.released.set();control.close()
