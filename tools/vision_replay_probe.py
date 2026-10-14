"""Exercise historical MCAP → ROS simulated clock → actual assistant image path.

Images are labelled synthetic640x480 blue RGB and aligned constant depth.
The live camera/domain and personal database are not used or modified.
"""
import argparse
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import CACHE,sha256,stamp,write_json
from tools.vision_bag import TOPICS,ros_command,environment,close_process

HISTORICAL_NS=1_600_000_000_000_000_000
DOMAIN=195
SOURCE_FILES=('app/adapters/vision.py','app/vision_frames.py','app/vision_snapshot.py',
    'app/session.py','configs/runtime.yaml','models/manifest.json','deploy/vision/bag/player.cpp')


def source_fingerprints():
    return {name:sha256(ROOT/name) for name in SOURCE_FILES}


def make_bag(directory):
    import rosbag2_py
    from sensor_msgs.msg import Image,CameraInfo
    from rclpy.serialization import serialize_message
    writer=rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(directory),storage_id='mcap',storage_preset_profile='zstd_fast'),
        rosbag2_py.ConverterOptions('cdr','cdr'))
    for index,topic in enumerate(TOPICS):
        writer.create_topic(rosbag2_py.TopicMetadata(id=index+1,name=topic,
            type='sensor_msgs/msg/Image' if index<2 else 'sensor_msgs/msg/CameraInfo',serialization_format='cdr'))
    for frame in range(300):
        capture=HISTORICAL_NS+frame*100_000_000
        for index,topic in enumerate(TOPICS):
            message=Image() if index<2 else CameraInfo()
            message.header.stamp.sec=capture//10**9;message.header.stamp.nanosec=capture%10**9
            message.header.frame_id='synthetic_replay_optical'
            message.width=640;message.height=480
            if index<2:
                message.encoding='rgb8' if index==0 else '16UC1';message.step=640*(3 if index==0 else 2)
                message.data=(bytes([0,0,255]) if index==0 else bytes([232,3]))*(640*480)
            else:
                message.k=[600.,0.,320.,0.,600.,240.,0.,0.,1.]
            writer.write(topic,serialize_message(message),capture+20_000_000+index*100_000)
    del writer


async def wait_for(predicate,description,seconds=15):
    deadline=time.monotonic()+seconds
    while not predicate():
        if time.monotonic()>deadline:raise TimeoutError(description)
        await asyncio.sleep(.02)


async def probe(args):
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo
    from rosbag2_interfaces.srv import Pause,Resume,Seek
    from loguru import logger
    from app.adapters.vision import RosVisionBridge
    from app.vision_frames import VisionUnavailable
    from app.vision_snapshot import VisionSnapshotProcessor
    from tools.vision_bridge_probe import Capture
    from pipecat.frames.frames import EndFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.workers.runner import WorkerRunner

    logger.remove();logger.add(sys.stderr,level='WARNING')
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE']='LOCALHOST'
    if args.run_dir.exists() or args.output.exists():raise RuntimeError('Use fresh probe paths')
    args.run_dir.mkdir(parents=True,mode=0o700)
    (args.run_dir/'probe-source.py').write_bytes(Path(__file__).read_bytes())
    make_bag(args.run_dir/'bag')
    report={'created_at':stamp(),'status':'RUNNING','scope':__doc__,'checks':{},'errors':[],
        'historical_start_ns':HISTORICAL_NS,'fixture_frames':300,'source_sha256':sha256(Path(__file__)),
        'artifacts':str(args.run_dir.resolve()),'with_actual_assistant':args.with_assistant,
        'limitations':'Syntheticrecordedpixels,notactualD435ibandwidth,sceneaccuracy,gesture/voicequality orfinalJTacceptance.'}
    report['source_fingerprints']=source_fingerprints()
    report['native_player_sha256']=sha256(Path('/home/nvidia/AI_Stack/builds/vision-bag/vision-bag-play'))
    write_json(args.output,report)
    checks=report['checks'];player=None;log=None;session=None;runner_task=None;lease=None
    bridge=wall=None;capture=None;pipeline_errors=[]
    resources=ExitStack()
    try:
        bridge=RosVisionBridge(domain_id=DOMAIN,use_sim_time=True).start()
        wall=RosVisionBridge(domain_id=DOMAIN).start()
        camera_info=[]
        info_subscription=bridge.node.create_subscription(CameraInfo,TOPICS[2],lambda m:camera_info.append(m.header.stamp.sec),qos_profile_sensor_data)
        async def command(kind,request=None):
            client=bridge.node.create_client(kind,'/rosbag2_player/'+kind.__name__.lower())
            try:
                await wait_for(client.service_is_ready,'native player service '+kind.__name__)
                future=client.call_async(request or kind.Request())
                await wait_for(future.done,'native service response')
                if future.exception():raise future.exception()
                return future.result()
            finally:bridge.node.destroy_client(client)

        if args.with_assistant:
            from app.adapters.llama import LlamaConfig
            from app.memory_runtime import open_memory
            from app.reminders import ReminderService
            from app.records import Records
            from app.session import AssistantSession
            settings=json.loads((ROOT/'configs/runtime.yaml').read_text())
            memory=resources.enter_context(open_memory(args.run_dir/'assistant/memory',settings=settings))
            reminders=resources.enter_context(ReminderService(args.run_dir/'assistant/reminders.sqlite'))
            records=resources.enter_context(Records(args.run_dir/'assistant/records.sqlite'))
            session=await AssistantSession(LlamaConfig.from_manifest(ROOT/'models/manifest.json'),memory,reminders,records,
                user_id='fixture-owner',vision_frames=bridge.frames,memory_top_k=settings['memory']['top_k'],
                memory_context_tokens=settings['memory']['context_token_limit']).start()

        def launch(label):
            output=(args.run_dir/(label+'.log')).open('w')
            native=['/home/nvidia/AI_Stack/builds/vision-bag/vision-bag-play',str(args.run_dir/'bag')]
            process=subprocess.Popen(ros_command(native),env=environment(DOMAIN),stdout=output,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            return process,output

        player,log=launch('play-first')
        await wait_for(lambda:bridge.status()['state']=='READY' and bridge.frames.counts['accepted']>=5,'historical RGB-D admission')
        await asyncio.sleep(.3)
        checks['wall_clock_bridge_rejects_history']=wall.frames.counts['accepted']==0 and wall.frames.counts['invalid_capture_clock_or_stale']>0
        lease=bridge.frames.acquire()
        snapshot=lease.validate()
        checks['historical_stamp_preserved']=HISTORICAL_NS<=snapshot.source_time_ns<HISTORICAL_NS+30*10**9
        checks['source_is_recorded_replay']=snapshot.source_kind=='recorded_replay'
        checks['rgb_depth_preserved']=snapshot.rgb==bytes([0,0,255])*(640*480) and snapshot.depth==bytes([232,3])*(640*480)
        checks['mapped_capture_is_fresh']=0<=time.monotonic_ns()-snapshot.capture_time_ns<200_000_000
        report['source_sample']={'source_timestamp_ns':snapshot.source_time_ns,'capture_time_ns':snapshot.capture_time_ns,
            'arrival_time_ns':snapshot.arrival_time_ns,'source_kind':snapshot.source_kind}
        lease.close();lease=None

        provider=VisionSnapshotProcessor(bridge.frames);capture=Capture(provider)
        worker=PipelineWorker(Pipeline([provider,capture]),enable_rtvi=False,idle_timeout_secs=None)
        started=asyncio.Event()
        @worker.event_handler('on_pipeline_started')
        async def on_started(_worker,_frame):started.set()
        @worker.event_handler('on_pipeline_error')
        async def error(_worker,frame):
            with bridge.frames.lock:
                timings=[bridge.frames.resident[k].capture_time_ns for k in bridge.frames.queue]
            pipeline_errors.append({'error':frame.error,'metadata':frame.metadata,
                'now_ns':time.monotonic_ns(),'frame_capture_times_ns':timings})
        runner=WorkerRunner(handle_sigint=False,handle_sigterm=False);await runner.add_workers(worker)
        runner_task=asyncio.create_task(runner.run())
        await asyncio.wait_for(started.wait(),5)
        for i in range(3):
            await worker.queue_frame(provider.request('recordedfixture',user_id='fixture-owner',turn_id=f'replay-{i}'))
            await wait_for(lambda:len(capture.results)==i+1 or bool(pipeline_errors),'native Pipecat image request')
            if pipeline_errors:raise RuntimeError('Pipecat image request failed')
            await asyncio.sleep(.1)
        checks['pipecat_replay_frames']=len(capture.results)==3 and not pipeline_errors
        report['pipecat_samples']=capture.results
        await worker.queue_frame(EndFrame());await asyncio.wait_for(runner_task,5);runner_task=None

        lease=bridge.frames.acquire();old_epoch=lease.frame.epoch
        await command(Pause);await asyncio.sleep(.7)
        try:
            unexpected=bridge.frames.acquire();unexpected.close();checks['pause_stale_refusal']=False
        except VisionUnavailable:checks['pause_stale_refusal']=bridge.status()['state']=='STALE'
        await command(Resume)
        await wait_for(lambda:bridge.status()['state']=='READY' and bridge.frames.epoch!=old_epoch,'resume with new epoch')
        try:lease.validate();checks['paused_lease_revoked_on_resume']=False
        except VisionUnavailable:checks['paused_lease_revoked_on_resume']=True
        lease.close();lease=None

        lease=bridge.frames.acquire();old_epoch=lease.frame.epoch
        seek=Seek.Request();seek.time.sec=HISTORICAL_NS//10**9+1
        await command(Seek,seek)
        await wait_for(lambda:bridge.frames.epoch!=old_epoch,'seek invalidates epoch')
        try:lease.validate();checks['seek_revokes_old_lease']=False
        except VisionUnavailable:checks['seek_revokes_old_lease']=True
        lease.close();lease=None
        await wait_for(lambda:bridge.status()['state']=='READY','fresh frames after seek')
        if session:
            emitted=[]
            async def emit(text):emitted.append(text)
            answer=await session.ask('这张录制画面主要是什么颜色？只说颜色。',emit=emit,visual=True)
            checks['actual_assistant_blue_answer']='蓝' in answer['response'] and bool(emitted)
            checks['assistant_source_and_usage']=answer['image']['source_kind']=='recorded_replay' and answer['prompt_tokens']==[r['prompt_tokens'] for r in answer['model_usage']]
            checks['assistant_has_no_image_leak']='image_url' not in str(session.bot.context.get_messages()) and bridge.frames.status()['leased_frames']==0
            report['assistant']={'answer':answer['response'],'image':answer['image'],'prompt_tokens':answer['prompt_tokens'],
                'model_usage':answer['model_usage'],'elapsed_seconds':answer['elapsed_seconds'],'calls':answer['calls']}

        seek=Seek.Request();seek.time.sec=HISTORICAL_NS//10**9+29
        await command(Seek,seek)
        await wait_for(lambda:player.poll() is not None,'natural replay EOF',seconds=8)
        checks['native_player_clean_eof']=player.returncode==0
        await asyncio.sleep(.7)
        try:
            unexpected=bridge.frames.acquire();unexpected.close();checks['eof_stale_refusal']=False
        except VisionUnavailable:checks['eof_stale_refusal']=bridge.status()['state']=='STALE'
        report['stream_before_restart']=bridge.status();old_epoch=bridge.frames.epoch
        log.close();player,log=launch('play-restarted')
        await wait_for(lambda:bridge.status()['state']=='READY' and bridge.frames.epoch!=old_epoch,'replay restart epoch')
        checks['restart_new_epoch']=True
        report['active']=bridge.status();report['wall_clock_bridge']=wall.status()
        checks['payload_budget']=report['active']['peak_payload_bytes']<=report['active']['max_payload_bytes']
        checks['no_ros_failure']=bridge.failure is None and wall.failure is None
    except Exception as error:
        report['errors'].append({'type':type(error).__name__,'error':str(error)})
        if bridge:report['failed_bridge_state']=bridge.status()
    finally:
        report['pipeline_errors']=pipeline_errors
        if capture:report['pipecat_samples']=capture.results
        if lease:lease.close()
        if session:await session.close()
        resources.close()
        if runner_task:
            await worker.queue_frame(EndFrame());await asyncio.wait_for(runner_task,5)
        close_process(player)
        if log:log.close()
        for value in (bridge,wall):
            if value:value.close()
        checks['closed_bridge_resources']=bridge is not None and bridge.thread is None and bridge.frames.status()['resident_payload_bytes']==0
        checks['source_unchanged']=source_fingerprints()==report['source_fingerprints']
        report.update(status='PASS' if not report['errors'] and all(checks.values()) else 'FAIL',completed_at=stamp())
        report['raw_artifacts']=[{'path':str(p.resolve()),'sha256':sha256(p),'bytes':p.stat().st_size}
            for p in args.run_dir.rglob('*') if p.is_file() and 'assistant' not in p.relative_to(args.run_dir).parts]
        write_json(args.output,report)
    print(json.dumps({'status':report['status'],'checks':checks,'errors':report['errors']},ensure_ascii=False),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--with-assistant',action='store_true')
    raise SystemExit(asyncio.run(probe(parser.parse_args())))
