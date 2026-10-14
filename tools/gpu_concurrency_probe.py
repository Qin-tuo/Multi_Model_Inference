"""Collect native CUDA timelines for the current VLM and two live ROS models.

This is an instrumented diagnostic, not an end-to-end performance acceptance.
The controller temporarily replaces idle managed llama/vision/pose processes
with the same commands beneath one Nsight process tree, then restores them.
Camera, ASR, TTS and storage remain running. No captured images are persisted.
"""
import argparse
import base64
from collections import Counter
from contextlib import contextmanager
import ctypes
import io
import json
import os
from pathlib import Path
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy import services
from tools.project import CACHE, meminfo, request_json, sha256, stamp, wait_ready, write_json

NSYS = Path('/opt/nvidia/nsight-systems/2026.3.1/target-linux-sbsa-armv8/nsys')
COMPONENTS = ['llama', 'vision', 'pose']


def group_members(pgid):
    rows = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            data = path.read_text().rsplit(')', 1)[1].split()
            if int(data[2]) == pgid:
                rows.append({'pid': int(path.parent.name), 'start_ticks': int(data[19]), 'state': data[0],
                    'command': (path.parent / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')})
        except (OSError, ValueError):
            continue
    return rows


def terminate_group(process):
    # ROS launchers can exit before their native child. Always check the group.
    def remaining():
        process.poll()  # Reap the direct child before touching adopted grandchildren.
        for row in group_members(process.pid):
            if row['pid'] == process.pid:
                continue
            try:
                os.waitpid(row['pid'], os.WNOHANG)
            except ChildProcessError:
                pass
        return group_members(process.pid)

    if group_members(process.pid):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while remaining() and time.monotonic() < deadline:
        time.sleep(.1)
    if remaining():
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)
    deadline = time.monotonic() + 5
    while remaining() and time.monotonic() < deadline:
        time.sleep(.1)
    if remaining():
        raise RuntimeError('Profiled process group did not exit')


def adopt_children():
    # Linux PR_SET_CHILD_SUBREAPER. The benchmark owns the temporary process
    # tree; adopt/reap ROS grandchildren if ros2 exits first during shutdown.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot adopt benchmark descendants')


def cleanup_recorded(report):
    """On abnormal profiler exit, signal only still-identical process groups."""
    errors = []
    for name, row in reversed(list(report.get('processes', {}).items())):
        current = group_members(row['leader_pid'])
        if not current:
            continue
        recorded = {(r['pid'], r['start_ticks']) for r in row.get('members', [])}
        if not recorded or any((r['pid'], r['start_ticks']) not in recorded for r in current):
            errors.append('Cannot prove ownership of remaining group: ' + name)
            continue
        for sig, seconds in ((signal.SIGTERM,15), (signal.SIGKILL,5)):
            live = [r for r in group_members(row['leader_pid']) if r['state'] != 'Z']
            if not live:
                break
            os.killpg(row['leader_pid'],sig)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and any(r['state'] != 'Z' for r in group_members(row['leader_pid'])):
                time.sleep(.1)
        if any(r['state'] != 'Z' for r in group_members(row['leader_pid'])):
            errors.append('Profiled group still alive: ' + name)
    return errors


def fingerprints():
    paths = [ROOT / name for name in ('configs/runtime.yaml', 'models/manifest.json',
        'deploy/upstreams.lock.json', 'deploy/vision/source-lock.json', 'tools/project.py',
        'deploy/vision/serve.py', 'tools/vision_runtime_probe.py', 'tools/gpu_concurrency_probe.py')]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def inventory():
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    rows = []
    for item in manifest['models']['vlm']['files']:
        path = CACHE / 'models' / item['name']
        digest = sha256(path)
        if digest != item['sha256'] or path.stat().st_size != item['size']:
            raise RuntimeError('VLM artifacts differ from their lock')
        rows.append({'component': 'llama', 'path': str(path.resolve()),
            'sha256': digest, 'bytes': path.stat().st_size})
    for name in ('detection', 'pose'):
        item = manifest['models'][name]['onnx']
        path = CACHE / 'models/yolo' / item['name']
        if sha256(path) != item['sha256']:
            raise RuntimeError('Vision ONNX differs from its lock')
        paths = [path, *sorted((CACHE / 'models/yolo/ort-trt-fp32' / name).rglob('*.engine'))]
        if len(paths) != 2:
            raise RuntimeError('Expected one cached native TensorRT engine per model')
        rows += [{'component': name, 'path': str(p.resolve()), 'sha256': sha256(p),
                  'bytes': p.stat().st_size} for p in paths]
    return rows


def worker(args):
    import rclpy
    from PIL import Image
    from sensor_msgs.msg import Image as RosImage
    from rclpy.qos import qos_profile_sensor_data
    from tools.vision_runtime_probe import Probe
    from benchmarks.metrics import sampled_age_summary

    directory = args.run_dir
    adopt_children()
    report = {'created_at': stamp(), 'status': 'RUNNING', 'worker_pid':os.getpid(),
        'processes': {}, 'requests': [], 'errors': []}
    output = directory / 'worker.json'
    write_json(output, report)
    processes, logs = {}, []
    node = spin = None
    stop_spin = threading.Event()
    nvtx = ctypes.CDLL(str(directory / 'ranges.so'))
    nvtx.range_push.argtypes = [ctypes.c_char_p]
    nvtx.range_pop.argtypes = []

    @contextmanager
    def marked(label):
        nvtx.range_push(label.encode())
        try:
            yield
        finally:
            nvtx.range_pop()

    def interrupted(signum, _frame):
        raise InterruptedError(f'Profile worker received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for name in COMPONENTS:
            if meminfo()['MemAvailable'] < 2 * 1024**3:
                raise RuntimeError('Insufficient memory to start profiled model')
            log = (directory / (name + '.log')).open('w')
            logs.append(log)
            command = ['/usr/bin/python3', str(ROOT / 'tools/project.py'), 'serve', '--component', name]
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes[name] = process
            report['processes'][name] = {'leader_pid': process.pid, 'command': command,
                'members':group_members(process.pid)}
            write_json(output, report)
            if name == 'llama':
                wait_ready(process, 'http://127.0.0.1:8088', seconds=120)
                if not services.owns_port('llama', [process.pid]):
                    raise RuntimeError('Profiled VLM does not own its endpoint')
            else:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Profiled ROS model exited: ' + name)
                    if services.probe(name)['ready']:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('Profiled ROS model not ready: ' + name)
            report['processes'][name]['members'] = group_members(process.pid)
            write_json(output, report)

        rclpy.init()
        node = Probe()
        latest = []
        node.create_subscription(RosImage, '/camera/camera/color/image_raw',
            lambda message: latest.__setitem__(slice(None), [message]), qos_profile_sensor_data)
        spin_errors = []

        def run_spin():
            try:
                while not stop_spin.is_set():
                    rclpy.spin_once(node, timeout_sec=.05)
            except Exception as error:
                spin_errors.append(str(error))

        spin = threading.Thread(target=run_spin, daemon=True)
        spin.start()
        deadline = time.monotonic() + 20
        while not latest or not all(row['latest'] is not None for row in node.rows.values()):
            if spin_errors or time.monotonic() > deadline:
                raise RuntimeError('ROS sampling prerequisites failed')
            time.sleep(.05)
        node.measuring = True
        started = time.monotonic_ns()
        with marked('assistant:measurement'):
            with marked('assistant:visual_only'):
                time.sleep(5)
            for index in range(6):
                content = '用两句话解释本地人工智能助手如何协助日常工作。'
                row = {'id': index, 'kind': 'text' if index % 2 == 0 else 'image'}
                if row['kind'] == 'image':
                    message = latest[0]
                    source_ns = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
                    age = (node.get_clock().now().nanoseconds - source_ns) / 1e6
                    if not 0 <= age < 500 or message.encoding not in ('rgb8', 'bgr8'):
                        raise RuntimeError('Camera snapshot is stale or unsupported')
                    raw = 'RGB' if message.encoding == 'rgb8' else 'BGR'
                    image = Image.frombytes('RGB', (message.width, message.height),
                        bytes(message.data), 'raw', raw, message.step)
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=75)
                    content = [{'type': 'text', 'text': '用一句中文描述摄像头画面。'},
                        {'type': 'image_url', 'image_url': {'url':
                            'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()}}]
                    row['camera'] = {'source_timestamp_ns': source_ns, 'age_ms': age,
                        'width': message.width, 'height': message.height}
                body = {'model': 'assistant-vlm', 'messages': [{'role': 'user', 'content': content}],
                    'temperature': 0, 'seed': 42, 'max_tokens': 64, 'cache_prompt': False, 'stream': False}
                with marked(f'assistant:request:{index}:{row["kind"]}'):
                    row['started_monotonic_ns'] = time.monotonic_ns()
                    response = request_json('http://127.0.0.1:8088/v1/chat/completions', body, timeout=45)
                    row['finished_monotonic_ns'] = time.monotonic_ns()
                # Do not publish captured scene descriptions, pixels or prompts with image data.
                row.update(usage=response.get('usage'), timings=response.get('timings'),
                    answer_characters=len(response['choices'][0]['message'].get('content') or ''),
                    elapsed_ms=(row['finished_monotonic_ns'] - row['started_monotonic_ns']) / 1e6)
                report['requests'].append(row)
                write_json(output, report)
                print(json.dumps({'request': index, 'kind': row['kind'], 'elapsed_ms': row['elapsed_ms']}), flush=True)
            with marked('assistant:visual_tail'):
                time.sleep(3)
        elapsed = (time.monotonic_ns() - started) / 1e9
        stop_spin.set()
        spin.join(5)
        if spin.is_alive() or spin_errors:
            raise RuntimeError('ROS sampler failed or did not stop')
        report['visual_streams'] = {name: {'received': len(row['arrivals']),
            'effective_hz': len(row['arrivals']) / elapsed,
            **sampled_age_summary(row['sampled_ages'], row['missing']),
            'invalid_clock': row['invalid_clock']} for name, row in node.rows.items()}
        report['measurement_seconds'] = elapsed
        report['status'] = 'PASS' if len(report['requests']) == 6 and all(
            r['answer_characters'] > 0 for r in report['requests']) else 'FAIL'
    except Exception as error:
        report['errors'].append({'type': type(error).__name__, 'error': str(error)})
        report['status'] = 'FAIL'
    finally:
        stop_spin.set()
        if spin is not None:
            spin.join(5)
        if node is not None and (spin is None or not spin.is_alive()):
            node.destroy_node()
            rclpy.shutdown()
        for name, process in reversed(list(processes.items())):
            try:
                terminate_group(process)
                report['processes'][name]['cleaned_up'] = True
                report['processes'][name]['exit_code'] = process.returncode
            except Exception as error:
                report['errors'].append({'stage': 'cleanup', 'component': name, 'error': str(error)})
                report['status'] = 'FAIL'
        for log in logs:
            log.close()
        report['completed_at'] = stamp()
        write_json(output, report)
    return 0 if report['status'] == 'PASS' else 1


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def overlap_ns(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path, worker_report):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        processes = {r['globalPid']: r['pid'] for r in connection.execute('SELECT * FROM PROCESSES')}
        pid_names = {member['pid']: name for name, row in worker_report['processes'].items()
                     for member in row.get('members', [])}
        kernels = [dict(r) for r in connection.execute(
            'SELECT start,end,deviceId,globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL')]
        for row in kernels:
            row['pid'] = processes.get(row['globalPid'])
            row['component'] = pid_names.get(row['pid'])
        ranges = [dict(r) for r in connection.execute(
            'SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE \'assistant:%\' AND end IS NOT NULL')]
        phases = []
        for window in ranges:
            start, end = window['start'], window['end']
            measured = [r for r in kernels if r['start'] < end and r['end'] > start]
            timelines = {name: union((max(start, r['start']), min(end, r['end']))
                for r in measured if r['component'] == name) for name in COMPONENTS}
            phases.append({'label': window['text'], 'start_ns': start, 'end_ns': end,
                'duration_ms': (end-start)/1e6,
                'components': {name: {'kernel_count': sum(r['component'] == name for r in measured),
                    'kernel_interval_union_ms': sum(b-a for a,b in times)/1e6} for name,times in timelines.items()},
                'pairwise_kernel_interval_overlap_ms': {a+'+'+b: overlap_ns(timelines[a],timelines[b])/1e6
                    for a,b in [('llama','vision'),('llama','pose'),('vision','pose')]}})
        diagnostics = [dict(r) for r in connection.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
        return {'kernel_count': len(kernels), 'kernel_pids': sorted({r['pid'] for r in kernels if r['pid'] is not None}),
            'unmapped_kernel_count': sum(r['component'] is None for r in kernels),
            'device_ids': sorted({r['deviceId'] for r in kernels}), 'phases': phases,
            'diagnostic_messages': dict(Counter(r['text'] for r in diagnostics)),
            'diagnostic_severity_counts': dict(Counter(r['severity'] for r in diagnostics)),
            'kernel_trace_errors': dict(Counter(r['text'] for r in diagnostics if r['severity'] == 3
                and r['text'] != 'CUDA device 0: Unified Memory trace is not supported by the current driver version or configuration.')),
            'excluded_error_scope': 'Unsupported Unified Memory tracing is retained in diagnostics; memory-fault metrics were not requested and are not claimed.',
            'trace_method': 'CUPTI software instrumentation; hardware tracing availability is recorded in native diagnostics',
            'scope': 'CUPTI GPU kernel intervals clipped to NVTX request ranges; unions avoid double-counting concurrent streams.'
                ' Overlapping kernel lifetimes do not establish simultaneous SM instruction execution (preemption may occur).'
                ' No GPU occupancy, scheduler-residency or slowdown causal claim.'}


def controller(args):
    from app.adapters.llama import provider_slots_idle
    if args.run_dir.exists() or args.output.exists():
        raise RuntimeError('Use new experiment paths and preserve earlier evidence')
    before = services.state(list(services.ORDER), health=True)
    if before['backend_status'] != 'READY':
        raise RuntimeError('All managed backends must be ready')
    if not provider_slots_idle(request_json('http://127.0.0.1:8088/slots', timeout=5)):
        raise RuntimeError('Managed VLM is busy')
    files = inventory()
    args.run_dir.mkdir(parents=True, mode=0o700)
    (args.run_dir / 'probe-source.py').write_bytes(Path(__file__).read_bytes())
    header = NSYS.parent / 'nvtx/include'
    source = args.run_dir / 'ranges.c'
    source.write_text('#include <nvtx3/nvToolsExt.h>\n'
        'void range_push(const char *s) { nvtxRangePushA(s); }\n'
        'void range_pop(void) { nvtxRangePop(); }\n')
    subprocess.run(['cc', '-shared', '-fPIC', '-O2', '-I', str(header), str(source),
        '-ldl', '-o', str(args.run_dir / 'ranges.so')], check=True, timeout=30)
    report = {'created_at': stamp(), 'status': 'RUNNING', 'scope': __doc__,
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'source_fingerprints': fingerprints(), 'verified_models_engines': files,
        'profiler_version': subprocess.check_output([str(NSYS),'--version'], text=True).strip(),
        'backends_before': before, 'errors': [], 'artifacts': str(args.run_dir.resolve())}
    write_json(args.output, report)
    script = ' && '.join('source ' + shlex.quote(str(p)) for p in (
        Path('/opt/ros/jazzy/setup.bash'), CACHE / 'workspaces/vision/install/setup.bash'))
    script += ' && exec ' + shlex.join(['/usr/bin/python3', str(Path(__file__).resolve()),
        '--worker', '--run-dir', str(args.run_dir.resolve())])
    command = [str(NSYS), 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
        '--cuda-graph-trace=node', '--wait=all', '--export=sqlite',
        '--output=' + str(args.run_dir / 'timeline'), '/bin/bash', '-c', script]
    report['command'] = command
    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID='42', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST', PYTHONUNBUFFERED='1')
    env.pop('ROS_LOCALHOST_ONLY', None)
    stopped = False
    profiler = None
    try:
        stopped = True
        report['stopped'] = services.stop(COMPONENTS)
        write_json(args.output, report)
        with (args.run_dir / 'profiler.log').open('w') as log:
            profiler = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            code = profiler.wait(timeout=600)
            report['profiler_exit_code'] = code
            if code != 0:
                report['errors'].append({'stage':'profiler','error':'Nsight or workload exited with code ' + str(code)})
        report['worker'] = json.loads((args.run_dir / 'worker.json').read_text())
        report['trace'] = analyze(args.run_dir / 'timeline.sqlite', report['worker'])
    except Exception as error:
        report['errors'].append({'stage': 'trace', 'type': type(error).__name__, 'error': str(error)})
    finally:
        if profiler is not None and profiler.poll() is None:
            try:
                terminate_group(profiler)
            except Exception as error:
                report['errors'].append({'stage':'profiler_cleanup','error':str(error)})
        # Guard against PID reuse before cleaning an abnormally exited worker.
        worker_path = args.run_dir / 'worker.json'
        if worker_path.exists():
            report['worker'] = json.loads(worker_path.read_text())
        for error in cleanup_recorded(report.get('worker',{})):
            report['errors'].append({'stage':'cleanup','error':error})
        remaining = [r for row in report.get('worker',{}).get('processes',{}).values()
                     for r in group_members(row['leader_pid']) if r['state'] != 'Z']
        if remaining:
            report['errors'].append({'stage':'cleanup', 'error':'Profile worker left processes', 'remaining':remaining})
        elif stopped:
            try:
                report['restored'] = services.start(COMPONENTS)
            except Exception as error:
                report['errors'].append({'stage': 'restore', 'error': str(error)})
        report['backends_after'] = services.state(list(services.ORDER), health=True)
        requests = [r for r in report.get('trace',{}).get('phases',[]) if r['label'].startswith('assistant:request:')]
        checks = {'no_errors': not report['errors'], 'worker_pass': report.get('worker',{}).get('status') == 'PASS',
            'six_traced_requests': len(requests) == 6,
            'each_request_has_three_gpu_models': bool(requests) and all(
                all(c['kernel_count'] > 0 for c in r['components'].values()) for r in requests),
            'all_kernels_mapped': report.get('trace',{}).get('unmapped_kernel_count') == 0,
            'one_gpu': len(report.get('trace',{}).get('device_ids',[])) == 1,
            'no_kernel_trace_errors': report.get('trace',{}).get('kernel_trace_errors') == {},
            'source_unchanged': fingerprints() == report['source_fingerprints'],
            'models_engines_unchanged': inventory() == files,
            'backends_ready_after': report['backends_after']['backend_status'] == 'READY',
            'other_services_not_restarted': all(before['components'][name]['main_pid'] ==
                report['backends_after']['components'][name]['main_pid'] for name in services.ORDER if name not in COMPONENTS)}
        report.update(checks=checks, status='PASS' if all(checks.values()) else 'FAIL', completed_at=stamp())
        report['raw_artifacts'] = [{'path': str(p.resolve()), 'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(args.run_dir.iterdir()) if p.is_file()]
        write_json(args.output, report)
    print(json.dumps({'status':report['status'],'checks':checks,'errors':report['errors']}, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker and args.output is None:
        parser.error('--output is required for the controller')
    raise SystemExit(worker(args) if args.worker else controller(args))
