# SPDX-License-Identifier: Apache-2.0
"""Synthetic trajectories verify policy only, not real gesture accuracy."""
from copy import deepcopy
from dataclasses import replace
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from app.config import Config, GestureConfig
from app.gestures import GestureRecognizer, open_palm
from app.turn_control import TurnController


def hand(x=0,y=0,closed=False,handedness=.95):
    points=np.array([[0,80],[-13,65],[-28,45],[-48,26],[-65,10],
        [-25,10],[-28,-25],[-30,-45],[-32,-60],[0,0],[0,-35],[0,-65],[0,-80],
        [24,5],[28,-25],[32,-50],[35,-65],[45,20],[54,0],[62,-20],[70,-30]],dtype=float)
    if closed:points[[4,8,12,16,20]]=points[0]+[1,-1]
    points+=np.array([230+x,250+y])
    return {'landmarks':np.c_[points,np.zeros(21)].tolist(),'score':.99,'handedness_score':handedness}


def scene(index,stamp,hands=None,epoch='camera-one'):
    return {'ready':True,'hands':[hand()] if hands is None else hands,
            'frame':{'frame_id':index,'source_epoch':epoch,'received_monotonic':stamp,'width':640,'height':480,'time_kind':'synthetic_monotonic'}}


def feed(recognizer,xs,step=.25,start=1000,offset=0,**kwargs):
    events=[]
    for i,x in enumerate(xs):events+=recognizer.update(scene(offset+i,start+i*step,[hand(x=x,**kwargs)]),now=start+i*step)
    return events


def test_open_hand_predicate_is_rotation_scale_invariant_and_rejects_folds():
    points=np.array(hand()['landmarks'])
    assert open_palm(points) and not open_palm(hand(closed=True)['landmarks'])
    matrix=np.array([[0,-1],[1,0]])
    points[:,:2]=points[:,:2]@matrix*2+500
    assert open_palm(points)
    assert not open_palm(np.zeros((21,3)))


def test_stationary_palm_requires_distinct_frames_and_emits_once_per_raise():
    recognizer=GestureRecognizer(GestureConfig());same=scene(0,1000)
    for _ in range(20):assert not recognizer.update(same,now=1000)
    assert not feed(recognizer,[0]*3,start=1000.25,offset=1)
    event=recognizer.update(scene(4,1001),now=1001)
    assert len(event)==1 and event[0]['kind']=='stop' and event[0]['first_frame_id']==0
    assert not feed(recognizer,[0]*12,start=1001.25,offset=5)


def test_wave_reversals_greet_once_and_dont_cancel_when_hand_stays_up():
    recognizer=GestureRecognizer(GestureConfig())
    events=feed(recognizer,[0,50,100,50,0,50,100])
    assert [e['kind'] for e in events]==['wave']
    assert not feed(recognizer,[100]*12,start=1001.75,offset=7)
    feed(recognizer,[100]*3,start=1004.75,offset=19,closed=True)
    events=feed(recognizer,[100]*5,start=1005.5,offset=22)
    assert [e['kind'] for e in events]==['stop']


def test_one_way_motion_vertical_motion_and_short_hold_do_not_trigger():
    assert not feed(GestureRecognizer(GestureConfig()),[0,30,60,90,120])
    recognizer=GestureRecognizer(GestureConfig());events=[]
    for i,y in enumerate([0,30,60,30,0,30,60]):events+=recognizer.update(scene(i,1000+i*.2,[hand(y=y)]),now=1000+i*.2)
    assert not events
    assert not feed(GestureRecognizer(GestureConfig()),[0]*4)


@pytest.mark.parametrize('interruption',['occlusion','expired','epoch','out_of_order','gap'])
def test_interruption_cannot_count_as_a_continuous_hold(interruption):
    recognizer=GestureRecognizer(GestureConfig())
    assert not feed(recognizer,[0]*4)
    stamp=1001;epoch='camera-one';index=4
    if interruption=='occlusion':recognizer.update(scene(4,stamp,[]),now=stamp)
    elif interruption=='expired':recognizer.update(scene(4,stamp),now=stamp+1)
    elif interruption=='epoch':epoch='new-camera'
    elif interruption=='out_of_order':recognizer.update(scene(1,1000.1),now=stamp)
    elif interruption=='gap':stamp+=1
    assert not recognizer.update(scene(index+1,stamp+.25,epoch=epoch),now=stamp+.25)


def test_two_hands_keep_separate_tracks_despite_confidence_order_change():
    recognizer=GestureRecognizer(GestureConfig());events=[]
    for i in range(5):
        hands=[hand(x=-100,handedness=.05),hand(x=200,handedness=.95)]
        if i%2:hands.reverse()
        events+=recognizer.update(scene(i,1000+i*.25,hands),now=1000+i*.25)
    assert len(events)==2 and len({e['source_id'] for e in events})==2
    assert all(e['kind']=='stop' for e in events)


def test_ambiguous_crossing_does_not_merge_two_incomplete_histories():
    recognizer=GestureRecognizer(GestureConfig());events=[]
    for i,x in enumerate([-100,-60,-20,0]):
        events+=recognizer.update(scene(i,1000+i*.25,[hand(x=x),hand(x=100-x)]),now=1000+i*.25)
    assert not events
    assert not recognizer.update(scene(4,1001,[hand(x=50)]),now=1001)


def test_degenerate_clipped_low_confidence_and_excess_hands_are_not_actions():
    recognizer=GestureRecognizer(GestureConfig())
    bad=hand();bad['score']=.1
    clipped=hand(x=-300)
    degenerate=hand();degenerate['landmarks']=np.zeros((21,3)).tolist()
    for i in range(8):assert not recognizer.update(scene(i,1000+i*.25,[bad,clipped,degenerate]),now=1000+i*.25)
    assert recognizer._state['rejected_hands']=={'low_confidence':1,'clipped_hand':1,'small_hand':1}
    for i in range(8):assert not recognizer.update(scene(20+i,1003+i*.25,[hand() for _ in range(9)]),now=1003+i*.25)


def event(kind='wave',age=0,epoch='camera-one'):
    import uuid
    stamp=time.monotonic()-age
    return {'event_id':uuid.uuid4().hex,'kind':kind,'source_id':'anonymous-hand','source_epoch':epoch,
            'frame_id':10,'first_frame_id':1,'evidence_start_monotonic':stamp-1,
            'evidence_end_monotonic':stamp,'confirmed_monotonic':stamp,'expires_monotonic':stamp+.75}


class Model:
    def __init__(self):self.started=threading.Event();self.release=threading.Event();self.cancelled=False
    def generate_stream(self,*args,**kwargs):
        self.started.set();self.release.wait(2)
        yield '你好',{};yield '',{'done':True}
    def cancel(self,event):self.cancelled=True;self.release.set()
    def idle(self):return True


def wait_done(control):
    deadline=time.monotonic()+2
    while control.state()['active_turn'] and time.monotonic()<deadline:time.sleep(.01)
    assert control.state()['active_turn'] is None


def test_wave_admission_busy_mute_expiry_dedup_and_global_cooldown():
    model=Model();cfg=Config.load();sink=SimpleNamespace(send=lambda *a,**kw:None,ptt_active=False)
    control=TurnController(cfg,model,sink,SimpleNamespace(source_epoch='camera-one'));control.start()
    try:
        assert control.handle_gesture(event(age=2))['reason']=='expired_or_wrong_source'
        assert control.handle_gesture(event(epoch='old'))['reason']=='expired_or_wrong_source'
        cfg.audio.input_mode='device'
        assert control.handle_gesture(event())['reason']=='greeting_muted'
        cfg.audio.input_mode='file';wave=event()
        accepted=control.handle_gesture(wave);assert accepted['accepted'] and model.started.wait(1)
        assert control.handle_gesture(wave)['reason']=='duplicate'
        assert control.handle_gesture(event())['reason']=='busy'
        model.release.set();wait_done(control)
        assert control.state()['last_result']['gesture']['event_id']==wave['event_id']
        assert control.session_state()['retained_turns']==0
        assert control.handle_gesture(event())['reason']=='cooldown'
    finally:model.release.set();control.close()


def test_stop_uses_same_cancellation_even_when_muted_but_rejects_pre_turn_frame():
    model=Model();cfg=Config.load();cfg.audio.input_mode='device'
    control=TurnController(cfg,model,SimpleNamespace(send=lambda *a,**kw:None,ptt_active=False),SimpleNamespace(source_epoch='camera-one'))
    control.start()
    try:
        old=event('stop');key=control.submit('question')['turn_id'];assert model.started.wait(1)
        assert control.handle_gesture(old)['reason']=='frame_precedes_turn'
        stopped=control.handle_gesture(event('stop'))
        assert stopped['accepted'] and stopped['turn_id']==key
        wait_done(control)
        assert model.cancelled and control.state()['last_result']['status']=='cancelled'
        assert control.state()['last_result']['cancel_reason']=='stationary_palm'
    finally:model.release.set();control.close()
