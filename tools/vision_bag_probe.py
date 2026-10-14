"""Real rosbag2 MCAP record/read/replay with explicitly synthetic ROS messages."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import sha256,stamp,write_json
from tools.vision_bag import TOPICS,replay


def main(args):
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE']='LOCALHOST'
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.convert import message_to_ordereddict
    from sensor_msgs.msg import Image,CameraInfo
    import rosbag2_py

    def fingerprint(message):
        # CDR serializers may use different padding bytes. Compare every ROS
        # field, including original stamps, IDs, arrays and pixel bytes.
        return hashlib.sha256(json.dumps(message_to_ordereddict(message),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    if args.run_dir.exists() or args.output.exists():raise RuntimeError('Use fresh probe paths')
    args.run_dir.mkdir(parents=True,mode=0o700)
    context=Context();rclpy.init(context=context,domain_id=191)
    node=Node('assistant_bag_fixture_publisher',context=context)
    executor=SingleThreadedExecutor(context=context);executor.add_node(node)
    publishers=[node.create_publisher(Image if i<2 else CameraInfo,topic,qos_profile_sensor_data)
        for i,topic in enumerate(TOPICS)]
    commands=[sys.executable,str(ROOT/'tools/vision_bag.py'),'record','--output',str(args.run_dir/'clip'),
        '--seconds','5','--session-group','synthetic-session','--subject','fixture','--split','development',
        '--domain','191','--fixture']
    emitted=Counter();process=None
    try:
        with (args.run_dir/'record-controller.log').open('w') as log:
            process=subprocess.Popen(commands,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+20;sequence=0
            while process.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Record fixture deadline')
                if all(p.get_subscription_count()>0 for p in publishers):
                    header_stamp=node.get_clock().now().to_msg()
                    for i,publisher in enumerate(publishers):
                        message=Image() if i<2 else CameraInfo()
                        message.header.stamp=header_stamp
                        message.header.frame_id=f'SYNTHETIC_FIXTURE_{sequence}'
                        message.width=8;message.height=6
                        if i<2:
                            message.encoding='rgb8' if i==0 else '16UC1'
                            message.step=8*(3 if i==0 else 2)
                            message.data=bytes([sequence%200])*message.step*6
                        else:
                            message.k=[1.,0.,4.,0.,1.,3.,0.,0.,1.]
                        emitted[(TOPICS[i],fingerprint(message))]+=1
                        publisher.publish(message)
                    sequence+=1
                executor.spin_once(timeout_sec=.02);time.sleep(.05)
            if process.returncode!=0:raise RuntimeError('Recorder fixture failed')
    finally:
        if process is not None and process.poll() is None:
            process.terminate();process.wait(timeout=20)
        executor.shutdown();node.destroy_node();rclpy.shutdown(context=context)

    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.run_dir/'clip/bag'),storage_id='mcap'),
        rosbag2_py.ConverterOptions('cdr','cdr'))
    recorded=Counter();source_headers=[]
    while reader.has_next():
        topic,payload,stamp_ns=reader.read_next()
        message=deserialize_message(payload,CameraInfo if topic==TOPICS[2] else Image)
        recorded[(topic,fingerprint(message))]+=1
        source_headers.append(message.header.frame_id)
    del reader

    playback_context=Context();rclpy.init(context=playback_context,domain_id=192)
    consumer=Node('assistant_bag_fixture_consumer',context=playback_context)
    playback_executor=SingleThreadedExecutor(context=playback_context);playback_executor.add_node(consumer)
    received=Counter()
    def callback(topic):
        def accept(message):received[(topic,fingerprint(message))]+=1
        return accept
    subscriptions=[consumer.create_subscription(Image if i<2 else CameraInfo,topic,callback(topic),qos_profile_sensor_data)
        for i,topic in enumerate(TOPICS)]
    playback=None
    try:
        command=[sys.executable,str(ROOT/'tools/vision_bag.py'),'replay','--recording',str(args.run_dir/'clip'),'--domain','192']
        with (args.run_dir/'replay.log').open('w') as log:
            playback=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+40
            while playback.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Replay fixture deadline')
                playback_executor.spin_once(timeout_sec=.05)
            end=time.monotonic()+.5
            while time.monotonic()<end:playback_executor.spin_once(timeout_sec=.05)
            if playback.returncode!=0:raise RuntimeError('Replay fixture failed')
    finally:
        if playback is not None and playback.poll() is None:
            playback.terminate();playback.wait(timeout=20)
        playback_executor.shutdown();consumer.destroy_node();rclpy.shutdown(context=playback_context)
    try:
        replay(argparse.Namespace(recording=args.run_dir/'clip',domain=42))
        live_replay_rejected=False
    except ValueError:
        live_replay_rejected=True
    checks={'nonempty_each_topic':all(any(t==topic for t,_ in recorded) for topic in TOPICS),
        'all_recorded_fields_were_emitted':all(emitted[key]>=count for key,count in recorded.items()),
        'recorded_and_replayed_fields_identical':recorded==received,
        'synthetic_headers_preserved':bool(source_headers) and all(x.startswith('SYNTHETIC_FIXTURE_') for x in source_headers),
        'live_domain_replay_rejected':live_replay_rejected}
    report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'emitted_messages':sum(emitted.values()),'recorded_messages':sum(recorded.values()),'replayed_messages':sum(received.values()),
        'record_domain':191,'replay_domain':192,'source_sha256':sha256(Path(__file__)),
        'recorder_sha256':sha256(ROOT/'tools/vision_bag.py'),'artifacts':str(args.run_dir.resolve()),
        'native_player_source_sha256':sha256(ROOT/'deploy/vision/bag/player.cpp'),
        'native_player_binary_sha256':sha256(Path('/home/nvidia/AI_Stack/builds/vision-bag/vision-bag-play')),
        'scope':'Small8x6syntheticImage/CameraInfo messages throughactualMCAP/zstdrecording andROSreplay. All ROS fields compared,includingstamps,pixels,intrinsics;unspecifiedCDRpadding excluded. No real camera capture, bandwidth claim, human labels, assistant simulated-clock integration, accuracy orJTacceptance.'}
    write_json(args.output,report);print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
