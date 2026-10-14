"""Compare one runtime factor in the existing voice/memory/vision workload.

Clone sealed synthetic seeds before measurement. Restore the original runtime
configuration and managed model services in finally. The observer/GC/input
settings are identical. A completed workload can still FAIL latency gates.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from app.adapters.llama import provider_slots_idle
from deploy import services
from tools.project import sha256, stamp, write_json


def idle():
    with httpx.Client(timeout=5, trust_env=False) as client:
        for port in (8088, 8089):
            response = client.get(f'http://127.0.0.1:{port}/slots'); response.raise_for_status()
            if not provider_slots_idle(response.json()):
                raise RuntimeError(f'Model at {port} is busy')


def main(args):
    if args.run_dir.exists() or args.output.exists():
        raise RuntimeError('Use fresh experiment paths')
    runtime = ROOT / 'configs/runtime.yaml'
    original = runtime.read_bytes()
    json.loads(original)
    expected_current = original
    args.run_dir.mkdir(parents=True, mode=0o700)
    (args.run_dir / 'original-runtime.json').write_bytes(original)
    variants = [('baseline', {'type': 'none'}), ('ngram_simple', {
        'type': 'ngram-simple', 'match_tokens': 2, 'draft_tokens': 4, 'min_hits': 1})]
    if args.factor == 'embedding_threads':
        variants = [('baseline', {'embedding_threads': 2}), ('threads4', {'embedding_threads': 4})]
    report = {'created_at': stamp(), 'status': 'PREPARING', 'source_sha256': sha256(Path(__file__)),
        'original_runtime_sha256': sha256(runtime), 'seed_source': str(args.seed_source),
        'seed_snapshot_sha256': sha256(args.seed_source / 'seed-backup/manifest.json'),
        'variants': {}, 'errors': [], 'factor': args.factor,
        'scope': 'Two 20-turn fixed 10s file-input runs with real Pipecat/ASR/LLM/TTS,10k personal-memory queries/writes'
            ' and live camera/detection/pose. New identical snapshot clones,cold VLM/ASR/speech per variant;'
            ' startup collect and passive GC observer in both. Only the named runtime factor differs.'
            ' Live images/time answers may vary;5 samples per category,not physical audio,independent quality,B0-B3 or JT acceptance.'}
    write_json(args.output, report)
    stopped = False
    try:
        for label, policy in variants:
            folder = args.run_dir / label
            subprocess.run([sys.executable, str(ROOT / 'tools/clone_capacity_fixture.py'),
                '--source', str(args.seed_source), '--target', str(folder), '--from-snapshot'], check=True)
            report['variants'][label] = {'policy': policy, 'folder': str(folder), 'phase': 'PREPARED'}
            write_json(args.output, report)
        for label, policy in variants:
            row = report['variants'][label]
            idle()
            if runtime.read_bytes() != expected_current:
                raise RuntimeError('Runtime configuration changed outside this experiment')
            stopped = True
            row['stopped'] = services.stop(['llama', 'asr'])
            config = json.loads(original)
            if args.factor == 'embedding_threads':
                config['memory'].update(policy)
            else:
                config['llama']['speculative'] = policy
            expected_current = (json.dumps(config, ensure_ascii=False, indent=2) + '\n').encode()
            runtime.write_bytes(expected_current)
            row['runtime_sha256'] = sha256(runtime)
            row['started'] = services.start(['llama', 'asr', 'speech'])
            with httpx.Client(timeout=5, trust_env=False) as client:
                response = client.get('http://127.0.0.1:8088/slots'); response.raise_for_status()
                row['native_slots'] = response.json()
                enabled = config['llama'].get('speculative', {}).get('type', 'none') != 'none'
                if not row['native_slots'] or any(slot.get('speculative') is not enabled for slot in row['native_slots']):
                    raise RuntimeError('Actual native speculative state differs from requested policy')
            output = args.output.parent / (args.output.stem + '-' + label + '.json')
            command = [sys.executable, str(ROOT / 'tools/heap_voice_worker.py'), '--startup-heap', 'collect',
                '--observe-gc', '--profile', 'bge-int8-windows-bm25-v1', '--memory-top-k', '8',
                '--memory-token-limit', '768', '--skip-literal-arithmetic', '--turn-interval', '10',
                '--turns', '20', '--run-dir', row['folder'], '--output', str(output)]
            row.update(phase='MEASURING', command=command, report=str(output))
            report['status'] = 'RUNNING'
            write_json(args.output, report)
            with (args.run_dir / (label + '.log')).open('w') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1000)
            row['returncode'] = process.returncode
            if not output.exists():
                raise RuntimeError(f'{label} produced no workload report')
            measured = json.loads(output.read_text())
            if measured['embedding_runtime']['intra_op_threads'] != config['memory'].get('embedding_threads', 2):
                raise RuntimeError('Measured ORT thread count differs from requested configuration')
            row.update(phase='FINISHED', status=measured['status'], checks=measured['checks'],
                voice_latency=measured['voice_latency'], query_summary=measured['query_summary'],
                write_summary=measured['write_summary'], reader_fingerprint=measured['reader_fingerprint'],
                embedding_runtime=measured['embedding_runtime'])
            write_json(args.output, report)
            print(json.dumps({'variant': label, 'status': row['status'], 'voice_latency': row['voice_latency']}, ensure_ascii=False), flush=True)
            if not measured['checks'].get('workload_completed'):
                raise RuntimeError(f'{label} did not complete;do not compare a partial workload')
    except Exception as error:
        report['errors'].append({'type': type(error).__name__, 'error': str(error)})
    finally:
        if stopped:
            try:
                services.stop(['llama', 'asr'])
                if runtime.read_bytes() != expected_current:
                    raise RuntimeError('External config edits preserved;automatic restoration refused')
                runtime.write_bytes(original)
                report['restored'] = services.start(['llama', 'asr', 'speech'])
            except Exception as error:
                report['errors'].append({'stage': 'restore', 'error': str(error)})
        report['configuration_restored'] = runtime.read_bytes() == original
        report['backends_after'] = services.state(list(services.ORDER), health=True)
        complete = len(report['variants']) == 2 and all(row.get('phase') == 'FINISHED' for row in report['variants'].values())
        report['status'] = 'COMPLETED_COMPARISON' if complete and not report['errors'] and report['configuration_restored'] else 'INCOMPLETE'
        report['completed_at'] = stamp()
        write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'errors': report['errors'],
        'configuration_restored': report['configuration_restored']}), flush=True)
    return 0 if report['status'] == 'COMPLETED_COMPARISON' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factor', choices=['speculative', 'embedding_threads'], default='speculative')
    parser.add_argument('--seed-source', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
