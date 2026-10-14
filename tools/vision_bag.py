"""Bounded recording and isolated replay through installed ROS 2 rosbag2/MCAP.

This records explicitly requested scene clips. A completed recording is not a
ground-truth annotation, quality result, or proof of lossless camera capture.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import CACHE,sha256,stamp,write_json

TOPICS=['/camera/camera/color/image_raw','/camera/camera/aligned_depth_to_color/image_raw',
        '/camera/camera/aligned_depth_to_color/camera_info']


def environment(domain):
    if not 0<=domain<=232:raise ValueError('ROS domain must be0..232')
    env=os.environ.copy()
    env.update(ROS_DOMAIN_ID=str(domain),ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
    env.pop('ROS_LOCALHOST_ONLY',None)
    return env


def ros_command(command):
    setup=' && '.join('source '+shlex.quote(str(p)) for p in (
        Path('/opt/ros/jazzy/setup.bash'),CACHE/'workspaces/vision/install/setup.bash'))
    return ['bash','-c',setup+' && exec '+shlex.join(command)]


def interrupted(signum,_frame):
    raise InterruptedError(f'Controller received signal {signum}')


def close_process(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            return False
    return True


def record(args):
    if args.output.exists():raise RuntimeError('Use a new recording directory')
    if not 1<=args.seconds<=60:raise ValueError('A clip must be1..60seconds')
    if args.fixture and args.domain in {42,int(os.environ.get('ASSISTANT_ROS_DOMAIN_ID','42'))}:
        raise ValueError('Synthetic fixture recording requires an isolated domain')
    for value in (args.session_group,args.subject):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',value):raise ValueError('Use anonymous group/subject identifiers')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    required=int(args.seconds*64*1024**2)+1024**3
    if shutil.disk_usage(args.output.parent).free<required:raise RuntimeError('Insufficient disk headroom for a bounded clip')
    args.output.mkdir(mode=0o700)
    qos=args.output/'qos.json'
    write_json(qos,{topic:{'history':'keep_last','depth':10,'reliability':'best_effort','durability':'volatile'} for topic in TOPICS})
    native=['ros2','bag','record','--storage','mcap','--storage-preset-profile','zstd_fast',
        '--output',str(args.output/'bag'),'--topics',*TOPICS,
        '--max-cache-size',str(16*1024**2),'--max-bag-size',str(128*1024**2),
        '--qos-profile-overrides-path',str(qos),'--disable-keyboard-controls']
    report={'created_at':stamp(),'status':'RECORDING','session_group':args.session_group,
        'subject':args.subject,'split':args.split,'domain':args.domain,'requested_seconds':args.seconds,
        'topics':TOPICS,'source_sha256':sha256(Path(__file__)),
        'input_kind':'synthetic_transport_fixture' if args.fixture else 'explicit_scene_capture',
        'runtime_sha256':sha256(ROOT/'configs/runtime.yaml'),'model_manifest_sha256':sha256(ROOT/'models/manifest.json'),
        'scope':__doc__,'command':native,'errors':[]}
    output=args.output/'manifest.json';write_json(output,report)
    process=None
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with (args.output/'record.log').open('w') as log:
            started=time.monotonic()
            process=subprocess.Popen(ros_command(native),env=environment(args.domain),stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            report['recorder_pid']=process.pid;write_json(output,report)
            deadline=started+args.seconds
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Recorder exited before the requested stop')
                if shutil.disk_usage(args.output).free<1024**3:raise RuntimeError('Recording reached disk reserve')
                time.sleep(.1)
            os.killpg(process.pid,signal.SIGINT)
            process.wait(timeout=15)
            report['elapsed_seconds']=time.monotonic()-started
            if process.returncode!=0:raise RuntimeError('Recorder did not close cleanly')
        info=subprocess.run(ros_command(['ros2','bag','info',str(args.output/'bag')]),env=environment(args.domain),
            capture_output=True,text=True,check=True,timeout=15)
        (args.output/'info.txt').write_text(info.stdout)
        # Read native metadata, keeping user-provided labels separate.
        import yaml
        metadata=yaml.safe_load((args.output/'bag/metadata.yaml').read_text())['rosbag2_bagfile_information']
        counts={row['topic_metadata']['name']:row['message_count'] for row in metadata['topics_with_message_count']}
        report['topic_message_counts']=counts
        if any(counts.get(topic,0)<=0 for topic in TOPICS):raise RuntimeError('A required topic has no recorded messages')
        report['status']='RECORDED_UNANNOTATED'
    except Exception as error:
        report['errors'].append({'type':type(error).__name__,'error':str(error)})
        report['status']='FAIL'
    finally:
        if not close_process(process):
            report['errors'].append({'stage':'cleanup','error':'Forced recorder termination'})
            report['status']='FAIL'
        for sig,handler in previous.items():signal.signal(sig,handler)
        report['artifacts']=[{'path':str(p.relative_to(args.output)),'sha256':sha256(p),'bytes':p.stat().st_size}
            for p in sorted(args.output.rglob('*')) if p.is_file() and p!=output]
        report['completed_at']=stamp();write_json(output,report)
    print(json.dumps({'status':report['status'],'counts':report.get('topic_message_counts'),'errors':report['errors']}),flush=True)
    return 0 if report['status']=='RECORDED_UNANNOTATED' else 1


def replay(args):
    if args.domain in {42,int(os.environ.get('ASSISTANT_ROS_DOMAIN_ID','42'))}:
        raise ValueError('Replay must use an isolated domain, not the live assistant domain')
    report=json.loads((args.recording/'manifest.json').read_text())
    if report['status']!='RECORDED_UNANNOTATED':raise RuntimeError('Recording did not close successfully')
    for artifact in report['artifacts']:
        path=(args.recording/artifact['path']).resolve()
        if not path.is_relative_to(args.recording.resolve()) or sha256(path)!=artifact['sha256']:
            raise RuntimeError('Recording artifact differs from manifest')
    player=Path('/home/nvidia/AI_Stack/builds/vision-bag/vision-bag-play')
    if not player.is_file():raise RuntimeError('Build the native rosbag2 admission adapter first')
    command=[str(player),str(args.recording/'bag')]
    # Consumers needing freshness checks must use ROS simulated time. Historical
    # header stamps are deliberately preserved, never relabelled as live frames.
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    process=None
    try:
        process=subprocess.Popen(ros_command(command),env=environment(args.domain),
            stdin=subprocess.DEVNULL,start_new_session=True)
        return process.wait(timeout=report['requested_seconds']+30)
    finally:
        close_process(process)
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    capture=commands.add_parser('record')
    capture.add_argument('--output',type=Path,required=True)
    capture.add_argument('--seconds',type=float,required=True)
    capture.add_argument('--session-group',required=True)
    capture.add_argument('--subject',required=True)
    capture.add_argument('--split',choices=['development','holdout'],required=True)
    capture.add_argument('--domain',type=int,default=42)
    capture.add_argument('--fixture',action='store_true',help=argparse.SUPPRESS)
    play=commands.add_parser('replay');play.add_argument('--recording',type=Path,required=True)
    play.add_argument('--domain',type=int,required=True)
    args=parser.parse_args()
    raise SystemExit({'record':record,'replay':replay}[args.action](args))
