"""Compare completed, audited MMBench development runs without changing models."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.project import sha256, stamp, write_json


def read_run(prefix):
    paths = {name: ROOT / f'benchmarks/results/{prefix}-{name}.json'
             for name in ('inference', 'audit', 'score')}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    run, audit, score = (reports[name] for name in ('inference', 'audit', 'score'))
    assert run['status'] == 'ANSWERED' and all(run['checks'].values())
    assert audit['status'] == 'PASS' and all(audit['checks'].values())
    assert not audit['execution_errors']
    for result in (audit, score):
        assert result['inference_report_sha256'] == sha256(paths['inference'])
    assert score['base_questions'] == 100 and run['completed'] == score['variants']
    assert score['split'] == 'development'
    times = []
    tokens = []
    for artifact in score['answer_artifacts']:
        path = Path(artifact['path'])
        assert sha256(path) == artifact['sha256']
        row = json.loads(path.read_text())
        assert row['error'] is None and not row['unexpected_mutation'] and not row['image_leak']
        times.append(row['seconds'])
        tokens.extend(item['prompt_tokens'] for item in row['model_usage'])
    assert len(times) == run['completed']
    summary = {key: score[key] for key in ('canonical_accuracy', 'circular_accuracy', 'variant_accuracy',
        'unparsed', 'strict_letter_outputs', 'errors', 'unexpected_mutations', 'image_leaks',
        'screening_target_80_percent_met', 'categories')}
    summary.update(model=run['model_identity'], resources=run['resources'],
        prompt_tokens_min=min(tokens), prompt_tokens_max=max(tokens),
        wall_seconds={'median': statistics.median(times),
                      'p95_nearest_rank': sorted(times)[math.ceil(.95 * len(times)) - 1],
                      'max': max(times)},
        evidence={name: {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
                  for name, path in paths.items()})
    return run, score, summary


def main(args):
    if args.output.exists():
        raise RuntimeError('Preserve earlier comparisons; use a fresh output')
    base, base_score, base_summary = read_run('mmbench-cn-development')
    candidate, candidate_score, candidate_summary = read_run('mmbench-cn-qwen35')
    checks = {key: base[key] == candidate[key] for key in
              ('input_sha256', 'corpus_sha256', 'protocol_sha256', 'completed', 'split')}
    for key in ('corpus_sha256', 'protocol_sha256', 'scorer_lock_sha256'):
        checks['score_' + key] = base_score[key] == candidate_score[key]
    changed = subprocess.check_output(['git', 'diff', '--name-only', base['source_commit'],
        candidate['source_commit'], '--', 'app', 'configs/runtime.yaml', 'models/manifest.json'],
        cwd=ROOT, text=True).splitlines()
    checks['same_committed_application_runtime_and_default_model'] = not changed
    baseline = {row['base_id']: row for row in base_score['questions']}
    other = {row['base_id']: row for row in candidate_score['questions']}
    checks['same_question_groups'] = baseline.keys() == other.keys()
    assert all(checks.values()), checks
    paired = {}
    for metric in ('canonical_correct', 'circular_correct'):
        counts = {'both_correct': 0, 'candidate_only_correct': 0,
                  'baseline_only_correct': 0, 'both_wrong': 0}
        for key, before in baseline.items():
            after = other[key]
            assert (before['category'], before['variants']) == (after['category'], after['variants'])
            a, b = before[metric], after[metric]
            name = ('both_correct' if a and b else 'candidate_only_correct' if b
                    else 'baseline_only_correct' if a else 'both_wrong')
            counts[name] += 1
        counts['delta_percentage_points'] = counts['candidate_only_correct'] - counts['baseline_only_correct']
        paired[metric] = counts
    controller_path = ROOT / 'benchmarks/results/qwen35-mmbench-controller.json'
    controller = json.loads(controller_path.read_text())
    assert controller['status'] in {'PASS', 'FAIL'}
    assert controller['checks']['backends_ready_after'] and controller['checks']['active_manifest_unchanged']
    assert controller['checks']['worker_passed'] and not controller['errors']
    report = {'created_at': stamp(), 'status': 'DEVELOPMENT_COMPARISON', 'checks': checks,
        'source_sha256': sha256(Path(__file__)), 'baseline': base_summary, 'candidate': candidate_summary,
        'paired_questions': paired,
        'native_compatibility': {'status': controller['status'], 'passed': sum(c['expected_check'] for c in controller['cases']),
            'total': len(controller['cases']), 'failed_ids': [c['case_id'] for c in controller['cases'] if not c['expected_check']],
            'report': str(controller_path.relative_to(ROOT)), 'sha256': sha256(controller_path)},
        'restoration': {'default_manifest_unchanged': True, 'managed_backend_status': controller['backends_after']['backend_status']},
        'decision': 'Neither reader reaches the predeclared 80% development screening target. Keep default unchanged; do not consume heldout/reserve or claim final model selection.',
        'scope': 'Same 100 public development questions and 372 provided option rotations. Whole reader bundle differs, including tokenizer/template/projector. One run per reader; no confidence or causal parameter-count claim. Wall time includes independent session setup/cleanup and is not voice latency. Resource samples include resident services and candidate-controller overhead. No natural freeform QA, physical audio, or JT-07 acceptance claim.'}
    write_json(args.output, report)
    print(json.dumps({'paired': paired, 'baseline_wall': base_summary['wall_seconds'],
        'candidate_wall': candidate_summary['wall_seconds'], 'compatibility': report['native_compatibility']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
