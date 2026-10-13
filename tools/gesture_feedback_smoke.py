"""Real ROS subscription and TTS outbox checks; gesture inputs are labelled fixtures."""
import asyncio
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import uuid
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from loguru import logger
from sqlalchemy import select
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner
from rclpy.qos import qos_profile_sensor_data
from yolo_msgs.msg import DetectionArray, Detection, KeyPoint2D

from app.adapters.vision import RosVisionBridge
from app.adapters.speech import OVSTTSService
from app.notifications import NotificationPump, PipecatNotificationSink, NotificationFrameGate, NotificationOutputMixin
from app.speech_output import WavOutputDevice
from app.proactive import VisualFeedback
from app.reminders import ReminderService, _NOTIFICATIONS
from tools.project import DATA, stamp, write_json, sha256
from tools.voice_assistant_smoke import readback


class ProbeOutput(NotificationOutputMixin, WavOutputDevice):
    notification_channel='fixture'
    drain_notification_audio=WavOutputDevice.drain_notification_audio
    def __init__(self,path,sink):
        super().__init__(path,TransportParams(audio_out_enabled=True,audio_out_sample_rate=16000,
            audio_out_channels=1,audio_out_10ms_chunks=2,audio_out_end_silence_secs=0))
        self.bind_notifications(sink)
        self.first_pcm={}
        self.hold_drain=False
        self.at_drain=asyncio.Event()
        self.release_drain=asyncio.Event()
    async def drain_notification_audio(self):
        if self.hold_drain:
            self.at_drain.set()
            await self.release_drain.wait()
        await WavOutputDevice.drain_notification_audio(self)
    async def write_audio_frame(self,frame):
        item=self._notice_output
        now=time.monotonic_ns()
        written=await super().write_audio_frame(frame)
        if written and item: self.first_pcm.setdefault(item.notice['id'],now)
        return written


def pose(node,kind,step):
    message=DetectionArray()
    message.header.stamp=node.get_clock().now().to_msg()
    message.header.frame_id='SYNTHETIC_POSE_FIXTURE_OPTICAL'
    d=Detection(); d.class_id=0; d.class_name='person'; d.id='labelled-fixture-person'
    d.score=.95; d.bbox.center.position.x=320.; d.bbox.center.position.y=240.
    d.bbox.size.x=200.; d.bbox.size.y=360.
    d.bbox3d.frame_id='synthetic_camera_optical_frame'; d.bbox3d.center.position.z=1.5
    x=([0.,.3,-.3,.3,-.3][(step//2)%5] if kind=='wave' else 0.)
    for key,point in {6:(280,240),8:(280,205),10:(280+x*200,330 if kind=='neutral' else 170),
                      7:(360,240),9:(370,290),11:(380,330)}.items():
        p=KeyPoint2D(); p.id=key; p.point.x=float(point[0]); p.point.y=float(point[1]); p.score=.95
        d.keypoints.data.append(p)
    message.detections.append(d)
    return message


async def main():
    logger.remove(); logger.add(sys.stderr,level='WARNING')
    artifacts=DATA/'fixtures'/('gesture-feedback-'+uuid.uuid4().hex)
    artifacts.mkdir(parents=True)
    checks={}
    # Observe actual D435i/yolo_ros traffic without saving any images. Empty
    # scenes prove connectivity/freshness only, never positive-event accuracy.
    live=RosVisionBridge.configured().start()
    try:
        deadline=time.monotonic()+15
        while live.perception.counts['frames']<50 and time.monotonic()<deadline:
            if live.failure: raise RuntimeError('Live ROS adapter failed: '+live.failure)
            await asyncio.sleep(.1)
        live_state=live.status()
        checks['live_pose_subscription']=live_state['perception']['counts'].get('frames',0)>=50
        checks['live_rgbd_subscription']=live_state['counts'].get('accepted',0)>50
    finally: live.close()
    # Use an isolated domain, so synthetic messages cannot reach the actual
    # assistant's live scene or its personal data directory.
    bridge=RosVisionBridge(domain_id=187,perception=True).start()
    publisher=bridge.node.create_publisher(DetectionArray,'/pose/d435/detections_3d',qos_profile_sensor_data)
    state={'kind':'neutral','occupied':False,'publish':True}
    stop=asyncio.Event()
    async def feed():
        step=0
        while not stop.is_set():
            if state['publish']: publisher.publish(pose(bridge.node,state['kind'],step))
            step+=1
            await asyncio.sleep(.05)
    feeder=asyncio.create_task(feed())
    try:
        with tempfile.TemporaryDirectory(prefix='gesture-policy-') as temporary, ReminderService(Path(temporary)/'reminders.sqlite') as service:
            feedback=VisualFeedback(service,bridge.perception,'fixture-owner')
            sink=PipecatNotificationSink(service,occupied=lambda:state['occupied'],timeout_seconds=15)
            tts=OVSTTSService(notification_sink=sink)
            output=ProbeOutput(artifacts/'greetings.wav',sink)
            worker=PipelineWorker(Pipeline([NotificationFrameGate(sink),tts,output]),
                params=PipelineParams(audio_out_sample_rate=16000),idle_timeout_secs=None,enable_rtvi=False)
            sink.worker=worker
            runner=WorkerRunner(handle_sigint=False,handle_sigterm=False)
            await runner.add_workers(worker)
            runner_task=asyncio.create_task(runner.run())
            pump=NotificationPump(service,'fixture-owner',sink,occupied=lambda:state['occupied'],poll_seconds=.05)
            pump.before_claim=feedback.tick
            pump_task=asyncio.create_task(pump.run())
            def rows():
                with service.engine.connect() as c: return [dict(r) for r in c.execute(select(_NOTIFICATIONS)).mappings()]
            async def wait_for(predicate,seconds=10):
                async with asyncio.timeout(seconds):
                    while not predicate():
                        if bridge.failure: raise RuntimeError(bridge.failure)
                        if pump_task.done(): pump_task.result(); raise RuntimeError('Output pump stopped')
                        await asyncio.sleep(.02)
            try:
                await wait_for(lambda:bridge.perception.counts['frames']>=15)
                state['kind']='wave'
                await wait_for(lambda:len(pump.results)==1)
                checks['real_tts_file_delivery']=pump.results[0]['confirmed'] and rows()[0]['status']=='fixture_delivered'
                await asyncio.sleep(1)
                checks['held_wave_single_delivery']=len(rows())==1 and len(pump.results)==1
                feedback.set_muted(True)
                state['kind']='neutral'; await asyncio.sleep(.8)
                state['kind']='raise'; await asyncio.sleep(1.8)
                feedback.set_muted(False); await asyncio.sleep(.5)
                checks['muted_gesture_not_replayed']=len(rows())==1
                state['kind']='neutral'; await asyncio.sleep(.8)
                state['occupied']=True; state['kind']='raise'
                await wait_for(lambda:len(rows())==2)
                await asyncio.sleep(2.3)
                state['occupied']=False; await asyncio.sleep(.3)
                checks['foreground_event_expires']=rows()[1]['status']=='expired' and len(pump.results)==1
                state['kind']='neutral'; await asyncio.sleep(.8)
                output.hold_drain=True
                state['kind']='raise'
                await asyncio.wait_for(output.at_drain.wait(),10)
                state['publish']=False
                await asyncio.sleep(.7)
                output.release_drain.set()
                await wait_for(lambda:any(r['status']=='uncertain' for r in rows()))
                checks['stream_loss_before_delayed_drain_rejects_receipt']=rows()[-1]['status']=='uncertain'
                checks['no_physical_receipts']=all(r['played_utc'] is None for r in rows())
                checks['profile_isolation']=service.claim_next('different-fixture-profile') is None
                # The installed callback lacks a sender GID: graph ambiguity
                # must invalidate the scene rather than mix two trackers.
                extra=bridge.node.create_publisher(DetectionArray,'/pose/d435/detections_3d',qos_profile_sensor_data)
                try:
                    state['publish']=True
                    await wait_for(lambda:bridge.perception.counts['reset:ambiguous_pose_publisher']>0)
                    checks['multiple_publishers_rejected']=bridge.perception.snapshot()['state']=='STALE'
                finally:
                    state['publish']=False
                    bridge.node.destroy_publisher(extra)
                results=list(pump.results)
                notices=[{k:r[k] for k in ('id','status','output_channel','played_utc','delivery_attempts','delivery_evidence')} for r in rows()]
                final=feedback.status()
            finally:
                output.release_drain.set()
                await pump.stop(); await asyncio.wait_for(pump_task,20)
                await worker.queue_frame(EndFrame()); await asyncio.wait_for(runner_task,20)
                feedback.close()
    finally:
        stop.set(); await feeder; bridge.close()
    with wave.open(str(artifacts/'greetings.wav')) as wav:
        audio={'frames':wav.getnframes(),'rate':wav.getframerate(),'channels':wav.getnchannels()}
    transcript=await readback(artifacts/'greetings.wav')
    checks['audio_nonempty']=audio['frames']>16000
    checks['asr_readback_greeting']='你好' in transcript and '帮' in transcript
    report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_state':'working tree; see checkpoint containing this report',
        'scope':'live D435i/pose connectivity plus labelled synthetic ROS keypoint trajectories and real OVS TTS/ASR/Pipecat/WAV; source-loss case deliberately delays WAV drain; no physical audio or gesture recognition quality acceptance',
        'gesture_accuracy_acceptance':'NOT_VERIFIED','fixture_ros_domain':187,'live':live_state,
        'policy':final,'receipts':results,'notifications':notices,'output_audio':audio,
        'audio_readback':transcript,'audio_sha256':sha256(artifacts/'greetings.wav'),'artifacts':str(artifacts)}
    write_json(ROOT/'benchmarks/results/gesture-feedback-smoke.json',report)
    print(json.dumps({'status':report['status'],'checks':checks},ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__=='__main__': raise SystemExit(asyncio.run(main()))
