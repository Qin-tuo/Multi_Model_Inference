# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace
import json
import time
import numpy as np
import pytest
from app.camera import Camera, FrameSnapshot
from app.config import Config
from app.face_tracker import FaceTracker
from app.body_detector import BodyDetector, load_source
from app.perception import Perception, HandDetector


def frame(camera):
    image=np.zeros((100,200,3),np.uint8)
    return FrameSnapshot(7,camera.source_epoch,time.monotonic(),image,b'jpeg')


def test_body_preprocess_and_original_pixel_box_contract():
    image=np.zeros((100,200,3),np.uint8);image[:,:,0]=255
    blob,scale=BodyDetector.preprocess(image)
    assert blob.shape==(1,3,640,640) and scale==pytest.approx(200/640)
    assert blob[0,2,0,0]==1 and blob[0,0,0,0]==0
    detector=BodyDetector.__new__(BodyDetector)
    detector.entry={'output_shape':[1,84,8400],'class_id':0}
    detector.threshold=.5;detector.iou=.45;detector.max_detections=10
    output=np.zeros((1,84,8400),np.float32)
    output[0,:4,0]=[320,160,64,32];output[0,4,0]=.9
    result=detector.postprocess(output,scale)
    assert result[0]['type']=='person' and result[0]['box_xyxy']==pytest.approx([90,45,110,55])
    output[0,5,0]=.99
    assert detector.postprocess(output,scale)==[]


def test_palm_patch_keeps_two_nonoverlapping_palms():
    entry=json.loads(__import__('pathlib').Path("03_models/perception.json").read_text())['palm']
    source=entry['files'][entry['runtime_source_file']]
    module=load_source(source['path'],source['sha256'],'palm_nms_contract')
    model=module.MPPalmDet.__new__(module.MPPalmDet)
    model.input_size=np.array([192,192]);model.anchors=np.array([[120/640,120/640],[165/640,120/640]])
    model.score_threshold=.5;model.nms_threshold=.3;model.topK=5000
    box=np.zeros((1,2,18),np.float32);box[0,:,2:4]=12
    output=model._postprocess([box,np.array([[[10],[9]]],np.float32)],np.array([640,640]),np.array([0,0]))
    assert output.shape==(2,19)
    assert output[:,:4]==pytest.approx(np.array([[100,100,140,140],[145,100,185,140]]))


def test_hand_contract_preserves_all_21_points_and_declares_depth_kind():
    palm=np.zeros((1,19));palm[0,:4]=[10,10,50,50];palm[0,-1]=.9
    output=np.zeros(132);output[:4]=[5,5,60,60];output[4:67]=np.arange(63);output[130:]=[.7,.95]
    model=HandDetector.__new__(HandDetector);model.max_hands=2
    model.palm=SimpleNamespace(infer=lambda image:palm)
    model.hand=SimpleNamespace(infer=lambda image,roi:output)
    hands=model.infer(np.zeros((100,100,3),np.uint8))['hands']
    assert len(hands)==1 and len(hands[0]['landmarks'])==21
    assert hands[0]['landmarks'][20]==[60,61,62]
    assert hands[0]['landmark_confidence'] is None and hands[0]['z_kind']=='model_relative_wrist_pixels'


def test_one_original_frame_is_shared_by_all_models():
    camera=Camera(stale_after=1);sample=frame(camera);seen=[]
    class Detector:
        def detect(self,image):seen.append(id(image));return (80,30,120,70)
    tracker=FaceTracker(camera,Detector())
    perception=Perception(camera,Config.load().perception,tracker)
    class Body:
        provider='fake body'
        def infer(self,image):seen.append(id(image));return [{'type':'person','box_xyxy':[0,0,200,100],'score':.9}]
    class Hands:
        provider='fake hands'
        def infer(self,image):seen.append(id(image));return {'hands':[],'palms_detected':0}
    perception.body=Body();perception.hands=Hands()
    result=perception.process_frame(sample)
    assert seen==[id(sample.bgr)]*3
    assert result['frame']['frame_id']==result['face']['frame']['frame_id']==7
    assert result['face']['face_detected'] and result['persons'][0]['type']=='person'


def test_results_from_an_old_camera_epoch_are_rejected():
    camera=Camera();sample=frame(camera);perception=Perception(camera,Config.load().perception)
    class Body:
        provider='fake body'
        def infer(self,image):camera.source_epoch='new';return []
    perception.body=Body()
    with pytest.raises(RuntimeError,match='expired'):perception.process_frame(sample)


def test_expired_preview_cannot_keep_boxes_alive():
    camera=Camera(stale_after=.01);sample=frame(camera);perception=Perception(camera,Config.load().perception)
    perception._preview=sample;perception._state={'ready':True,'frame':sample.metadata()}
    time.sleep(.02)
    assert perception.preview()[0] is None
    assert not perception.snapshot()['ready']


@pytest.mark.parametrize('body_provider',['fake',None])
def test_vlm_image_and_auxiliary_observation_use_same_frame(body_provider):
    from app.turn_control import TurnController
    cfg=Config.load();camera=Camera();chosen=frame(camera)
    camera._alive=True;camera._latest=(time.monotonic(),chosen.bgr);camera._latest_jpeg=b'newer';camera._frame_id=8
    scene={'ready':True,'frame':chosen.metadata(),'persons':[{}],'hands':[{},{}],'face':{'face_detected':True},'providers':{'body':body_provider,'face':'fake','hands':'fake'}}
    class Model:
        def idle(self): return True
        def generate_stream(self,prompt,**kwargs):
            assert kwargs['images_b64']==[camera._base64_jpeg(chosen.jpeg)]
            assert 'person_candidates' in kwargs['system_prompt']
            yield '回答',{}
            yield '',{'done':True}
    sink=SimpleNamespace(send=lambda event,**kwargs:None)
    control=TurnController(cfg,Model(),sink,camera,perception=SimpleNamespace(preview=lambda:(chosen,scene)))
    try:
        control.start();control.submit('画面是什么',True)
        deadline=time.monotonic()+2
        while control.state()['active_turn'] and time.monotonic()<deadline:time.sleep(.01)
        result=control.state()['last_result']
        assert result['status']=='complete'
        assert result['frame']['frame_id']==result['perception_observation']['frame']['frame_id']==7
        assert result['perception_observation']['values']['person_candidates']==(1 if body_provider else None)
    finally:control.close();camera.close()
