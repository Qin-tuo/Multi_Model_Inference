"""Load a locked reader candidate in the real native server and restore VLM.

Use the existing synthetic text/tool/image cases,not held-out memory answers.
The active model manifest and runtime configuration are never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from app.adapters.llama import provider_slots_idle
from deploy import services
from deploy.resource_guard import hold_model_lease
from tools.llama_speculative_probe import body_for, request, output_ok
from tools.memory_capacity import MemoryMonitor
from tools.project import server_command, sha256, stamp, wait_ready, write_json


def main(args):
    if args.run_dir.exists() or args.output.exists():
        raise RuntimeError('Use new experiment paths')
    quality = (args.quality_run_dir, args.quality_output, args.seed_source)
    if any(quality) and not all(quality):
        raise RuntimeError('Quality run,output andseed source must be supplied together')
    if args.quality_run_dir and (args.quality_run_dir.exists() or args.quality_output.exists()):
        raise RuntimeError('Use new quality paths;preserve earlier answers')
    if bool(args.worker) != bool(args.worker_report):
        raise RuntimeError('A candidate worker requires its explicit report path')
    if args.worker_report and args.worker_report.exists():
        raise RuntimeError('Preserve earlier worker reports')
    candidate = json.loads(args.manifest.read_text())
    entry = candidate['models']['vlm']
    directory = Path(candidate['artifact_directory'])
    files = []
    for item in entry['files']:
        path = directory / item['name']
        if path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
            raise RuntimeError('Candidate artifact differs from its lock')
        files.append({'path': str(path), 'sha256': item['sha256'], 'bytes': item['size']})
    runtime = json.loads((ROOT / 'configs/runtime.yaml').read_text())
    if entry['context_tokens'] != runtime['llama']['context_tokens']:
        raise RuntimeError('This comparison must keep the current context capacity')
    corpus_path = ROOT / 'benchmarks/corpora/llama-speculative-v1.json'
    corpus = json.loads(corpus_path.read_text())
    before = services.state(list(services.ORDER), health=True)
    if before['backend_status'] != 'READY':
        raise RuntimeError('All managed backends must be ready before model screening')
    with httpx.Client(timeout=5, trust_env=False) as client:
        result = client.get('http://127.0.0.1:8088/slots'); result.raise_for_status()
        if not provider_slots_idle(result.json()):
            raise RuntimeError('Managed VLM is busy')
    args.run_dir.mkdir(parents=True, mode=0o700)
    command = server_command(model_entry=entry, model_directory=directory)
    report = {'created_at': stamp(), 'status': 'RUNNING', 'candidate_manifest': str(args.manifest),
        'candidate_manifest_sha256': sha256(args.manifest), 'source_sha256': sha256(Path(__file__)),
        'request_helper_sha256': sha256(ROOT / 'tools/llama_speculative_probe.py'),
        'active_manifest_sha256': sha256(ROOT / 'models/manifest.json'),
        'runtime_sha256': sha256(ROOT / 'configs/runtime.yaml'), 'verified_files': files,
        'command': command, 'corpus_sha256': sha256(corpus_path), 'backends_before': before,
        'artifacts': str(args.run_dir.resolve()), 'cases': [], 'errors': [],
        'scope': 'Real locked llama.cpp model/projector load plus14 fixed synthetic text/tool/image requests.'
            ' Native token-count preflight compared with usage;tools parsed,not executed.'
            ' Other backends stay resident/live;no foreground speech/memory load or visual rate measurement.'
            ' Compatibility screening,not independent quality,voice latency or JT acceptance.'}
    write_json(args.output, report)
    process = None
    stopped = False
    lease = hold_model_lease()
    try:
        stopped = True
        report['stopped_backend'] = services.stop(['llama'])
        with (args.run_dir / 'native.log').open('w') as log, MemoryMonitor() as monitor:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, pass_fds=(lease.fileno(),))
            report['candidate_pid'] = process.pid
            monitor.external_pid = process.pid
            wait_ready(process, 'http://127.0.0.1:8088', seconds=120)
            report['owned_listening_socket'] = services.owns_port('llama',[process.pid])
            if not report['owned_listening_socket']:
                raise RuntimeError('Candidate process does not own the model endpoint')
            with httpx.Client(timeout=60, trust_env=False) as client:
                response = client.get('http://127.0.0.1:8088/props'); response.raise_for_status()
                props = response.json()
                report['native_properties'] = {key: props[key] for key in (
                    'model_path','model_alias','modalities','total_slots','default_generation_settings','chat_template_caps','build_info') if key in props}
                if props.get('chat_template'):
                    report['native_chat_template_sha256'] = hashlib.sha256(props['chat_template'].encode()).hexdigest()
                for case in corpus['cases']:
                    body = body_for(case, entry)
                    count = client.post('http://127.0.0.1:8088/v1/chat/completions/input_tokens', json=body)
                    count.raise_for_status()
                    tokens = count.json()['input_tokens']
                    if type(tokens) is not int or tokens + entry['max_output_tokens'] + 16 > entry['context_tokens']:
                        raise RuntimeError('Candidate prompt exceeds the unchanged context budget')
                    row = request(client, body)
                    row.update(case_id=case['id'], kind=case['kind'], native_prompt_tokens=tokens,
                        token_count_matches_usage=tokens == row['usage']['prompt_tokens'], expected_check=output_ok(case,row))
                    report['cases'].append(row)
                    write_json(args.output, report)
                    print(json.dumps({'case': case['id'], 'expected': row['expected_check'],
                        'elapsed_ms': round(row['elapsed_ms'],2), 'tokens': tokens,
                        'reasoning_characters': row['reasoning_characters']}), flush=True)
                response = client.get('http://127.0.0.1:8088/slots'); response.raise_for_status()
                report['idle_after'] = provider_slots_idle(response.json())
            if args.quality_run_dir:
                # Exercise the unchanged AssistantSession with an explicitly
                # selected reader manifest and its verified native endpoint.
                quality_command = [sys.executable, str(ROOT / 'tools/lcvmem_eval.py'),
                    '--run-dir', str(args.quality_run_dir), '--output', str(args.quality_output),
                    '--seed-from', str(args.seed_source), '--split', 'development',
                    '--source-metadata', 'calendar-speaker-v1', '--memory-profile', 'bge-int8-windows-bm25-v1',
                    '--memory-top-k', '8', '--memory-token-limit', '768',
                    '--memory-evidence-format', args.evidence_format, '--model-manifest', str(args.manifest.resolve())]
                report['quality'] = {'status': 'RUNNING', 'command': quality_command}
                write_json(args.output, report)
                with (args.run_dir / 'quality.log').open('w') as log:
                    subprocess.run(quality_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                measured = json.loads(args.quality_output.read_text())
                report['quality'].update(status=measured['status'], report=str(args.quality_output),
                    questions=measured['selected'], f1_percent=measured['upstream_qa_f1_percent'],
                    errors=measured['errors'], mutations=measured['unexpected_mutations'],
                    owner_leaks=measured['cross_owner_evidence'])
                with httpx.Client(timeout=5, trust_env=False) as client:
                    response = client.get('http://127.0.0.1:8088/slots'); response.raise_for_status()
                    report['idle_after'] = provider_slots_idle(response.json())
            if args.worker:
                report['worker']={'status':'RUNNING','command':args.worker,'report':str(args.worker_report)}
                write_json(args.output,report)
                with (args.run_dir/'worker.log').open('w') as log:
                    subprocess.run(args.worker,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.worker_timeout)
                measured=json.loads(args.worker_report.read_text())
                report['worker'].update(status=measured['status'],checks=measured.get('checks'),
                    errors=measured.get('errors',0),timeout_seconds=args.worker_timeout)
                with httpx.Client(timeout=5,trust_env=False) as client:
                    response=client.get('http://127.0.0.1:8088/slots');response.raise_for_status()
                    report['idle_after']=provider_slots_idle(response.json())
            report['resources'] = monitor.report()
    except Exception as error:
        report['errors'].append({'stage': 'candidate', 'type': type(error).__name__, 'error': str(error)})
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=15)
            report['candidate_exit_code'] = process.returncode
        if stopped:
            try:
                report['restored_backend'] = services.start(['llama'])
            except Exception as error:
                report['errors'].append({'stage': 'restore_backend', 'error': str(error)})
        lease.close()
        report['backends_after'] = services.state(list(services.ORDER), health=True)
        checks = {'all_cases': len(report['cases']) == len(corpus['cases']), 'no_errors': not report['errors'],
            'all_expected_checks': bool(report['cases']) and all(r['expected_check'] for r in report['cases']),
            'native_preflight_matches_usage': bool(report['cases']) and all(r['token_count_matches_usage'] for r in report['cases']),
            'nonthinking_output': bool(report['cases']) and all(r['reasoning_characters'] == 0 for r in report['cases']),
            'idle_after': report.get('idle_after', False),
            'active_manifest_unchanged': report['active_manifest_sha256'] == sha256(ROOT / 'models/manifest.json'),
            'runtime_unchanged': report['runtime_sha256'] == sha256(ROOT / 'configs/runtime.yaml'),
            'backends_ready_after': report['backends_after']['backend_status'] == 'READY',
            'other_service_main_pids_unchanged': all(before['components'][name]['main_pid'] ==
                report['backends_after']['components'][name]['main_pid'] for name in services.ORDER if name != 'llama')}
        if args.quality_run_dir:
            result = report.get('quality', {})
            checks['quality_completed'] = result.get('status') == 'COMPLETE' and result.get('questions') == 82
            checks['quality_no_errors_or_mutations'] = all(result.get(key) == 0 for key in ('errors','mutations','owner_leaks'))
        if args.worker:
            result=report.get('worker',{})
            checks['worker_passed']=(result.get('status') in {'PASS','ANSWERED'} and result.get('errors',0) in (0,[])
                and all((result.get('checks') or {}).values()))
        report.update(checks=checks,status='PASS' if all(checks.values()) else 'FAIL',completed_at=stamp())
        write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'checks': checks, 'errors': report['errors']}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quality-run-dir', type=Path)
    parser.add_argument('--quality-output', type=Path)
    parser.add_argument('--seed-source', type=Path)
    parser.add_argument('--evidence-format', choices=['nested_v1','flat_v2'], default='flat_v2')
    parser.add_argument('--worker-report',type=Path)
    parser.add_argument('--worker-timeout',type=int,default=1800,choices=range(30,7201),metavar='SECONDS')
    parser.add_argument('--worker',nargs=argparse.REMAINDER,help='Explicit local validation command while candidate owns the endpoint')
    raise SystemExit(main(parser.parse_args()))
