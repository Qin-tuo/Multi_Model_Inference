"""Screen prompt batch sizes with pinned upstream llama-bench on the real GPU.

Temporarily replaces the idle managed VLM with the native benchmark, restores
it in finally, and observes the other live backends. No production settings,
models, prompts, or storage are changed. This is not end-to-end acceptance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from app.adapters.llama import provider_slots_idle
from benchmarks.metrics import sampled_age_summary
from deploy import services
from deploy.resource_guard import hold_model_lease
from tools.memory_capacity import MemoryMonitor
from tools.project import CACHE, sha256, stamp, write_json


def main(args):
    if args.output.exists() or args.run_dir.exists():
        raise RuntimeError('Use new report and artifact paths; keep earlier evidence')
    binary = CACHE / 'build/llama/bin/llama-bench'
    settings = json.loads((ROOT / 'configs/runtime.yaml').read_text())['llama']
    command = services.server_command()
    model = Path(command[command.index('-m') + 1])
    if not binary.is_file():
        raise RuntimeError('Build the pinned upstream llama-bench target first')
    before = services.state(list(services.ORDER), health=True)
    if before['backend_status'] != 'READY':
        raise RuntimeError('All seven managed backends must be ready before screening')
    with httpx.Client(timeout=5, trust_env=False) as client:
        response = client.get('http://127.0.0.1:8088/slots')
        response.raise_for_status()
        if not provider_slots_idle(response.json()):
            raise RuntimeError('The application VLM is busy')
    args.run_dir.mkdir(parents=True, mode=0o700)
    native_command = [str(binary), '-m', str(model), '-p', '512,1024,1536',
        '-n', '0', '-b', '2048', '-ub', '256,512,1024', '-ngl', str(settings['gpu_layers']),
        '-t', str(settings['threads']), '-fa', 'auto', '-r', '5', '-o', 'json', '--progress']
    report = {'created_at': stamp(), 'status': 'RUNNING', 'command': native_command,
        'native_binary_sha256': sha256(binary), 'model_sha256': sha256(model),
        'upstream_commit': subprocess.check_output(['git', '-C', str(CACHE / 'upstreams/llama.cpp'),
            'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': sha256(Path(__file__)), 'backends_before': before,
        'production_config_sha256': sha256(ROOT / 'configs/runtime.yaml'),
        'artifacts': str(args.run_dir.resolve()),
        'scope': 'Native llama-bench synthetic token prefill,5 repetitions per configuration;'
            ' no chat-template/tokenization/sampling,projector,ASR/TTS activity,memory load or'
            ' physical audio. Other backends remain loaded and camera/detection/pose live.'
            ' Native warmup is enabled. Screening only,not B0-B3 or JT acceptance.'}
    write_json(args.output, report)
    lease = hold_model_lease()
    stopped = False
    node = None
    observer = None
    observe_stop = threading.Event()
    errors = []
    try:
        # Own only this probe's ROS context. The managed camera/CV stay running.
        import rclpy
        from tools.vision_runtime_probe import Probe
        rclpy.init()
        node = Probe()
        ready_until = time.monotonic() + 20
        while not all(row['latest'] and time.monotonic_ns() - row['latest'] < 500_000_000
                      for row in node.rows.values()):
            if time.monotonic() > ready_until:
                raise RuntimeError('Live visual streams did not become ready')
            rclpy.spin_once(node, timeout_sec=.05)
        def observe():
            try:
                while not observe_stop.is_set():
                    rclpy.spin_once(node, timeout_sec=.05)
            except Exception as error:
                errors.append({'stage': 'vision_observer', 'error': str(error)})
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        # Set the restoration obligation before a stop that may partly succeed.
        stopped = True
        report['stopped_backend'] = services.stop(['llama'])
        write_json(args.output, report)
        started = time.monotonic_ns()
        node.started_ns = started
        node.measuring = True
        with MemoryMonitor() as monitor:
            with (args.run_dir / 'native.json').open('w') as out, (args.run_dir / 'native.log').open('w') as err:
                process = subprocess.run(native_command, stdout=out, stderr=err,
                    timeout=600, pass_fds=(lease.fileno(),))
            report['returncode'] = process.returncode
            report['resources'] = monitor.report()
        elapsed = (time.monotonic_ns() - started) / 1e9
        node.measuring = False
        observe_stop.set()
        observer.join(2)
        if observer.is_alive():
            raise RuntimeError('Visual observer did not stop')
        report['seconds'] = elapsed
        report['vision'] = {name: {'received': len(row['arrivals']),
            'effective_hz': len(row['arrivals']) / elapsed,
            **sampled_age_summary(row['sampled_ages'], row['missing']),
            'invalid_clock': row['invalid_clock']} for name, row in node.rows.items()}
        report['native_results'] = json.loads((args.run_dir / 'native.json').read_text())
    except Exception as error:
        errors.append({'stage': 'screening', 'type': type(error).__name__, 'error': str(error)})
    finally:
        observe_stop.set()
        if observer:
            observer.join(2)
        if node and not (observer and observer.is_alive()):
            node.destroy_node()
            rclpy.shutdown()
        if stopped:
            try:
                report['restored_backend'] = services.start(['llama'])
            except Exception as error:
                errors.append({'stage': 'restore_backend', 'error': str(error)})
        lease.close()
        report['backends_after'] = services.state(list(services.ORDER), health=True)
        report['errors'] = errors
        results = report.get('native_results', [])
        checks = {
            'native_completed': report.get('returncode') == 0 and len(results) == 9,
            'no_errors': not errors,
            'backends_ready_after': report['backends_after']['backend_status'] == 'READY',
            'other_backends_unchanged': all(before['components'][name]['cgroup_pids'] ==
                report['backends_after']['components'][name]['cgroup_pids'] for name in services.ORDER if name != 'llama'),
            'production_config_unchanged': report['production_config_sha256'] == sha256(ROOT / 'configs/runtime.yaml'),
            'vision_rates': all(report.get('vision', {}).get(name, {}).get('effective_hz', 0) >= 10
                for name in ('detection', 'pose')),
            'vision_freshness': all(report.get('vision', {}).get(name, {}).get('latest_age_p95_ms', float('inf')) <= 200
                for name in ('detection', 'pose')),
        }
        report.update(checks=checks, status='PASS' if all(checks.values()) else 'FAIL', completed_at=stamp())
        write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'checks': checks,
        'cases': [{k: row.get(k) for k in ('n_prompt', 'n_ubatch', 'avg_ts', 'stddev_ts')} for row in results]}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
