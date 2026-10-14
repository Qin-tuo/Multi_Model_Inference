"""Replay development retrieval snapshots through the real native tokenizer.

No questions/reference answers are used to select evidence. Reconstruct every
flat row into its nested representation to verify exact data conservation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.context_memory import memory_message
from tools.project import stamp, sha256, write_json


def payload(message):
    return json.loads(message['content'].split('\n', 1)[1])


def canonical(row, metrics):
    row = dict(row)
    if 'sources' not in row:
        mapped = metrics['evidence_references'][row['ref']]['sources']
        if len(mapped) != 1:
            raise RuntimeError('Flat row has an ambiguous source mapping')
        source = {key: row.pop(key) for key in ('date', 'speaker', 'evidence', 'span') if key in row}
        source['ref'] = next(iter(mapped))
        row['sources'] = [source]
    return row


def main(args):
    if args.output.exists():
        raise RuntimeError('Use a new report path')
    report = json.loads(args.source_report.read_text())
    if report['status'] != 'COMPLETE' or report['split'] != 'development' or report['selected'] != 82:
        raise RuntimeError('Only a complete 82-question development snapshot is supported')
    rows = []
    for case in report['cases']:
        raw_path = Path(report['raw_results_directory']) / (case['id'] + '.json')
        raw = json.loads(raw_path.read_text())
        items = raw['retrieved']
        before = json.dumps(items, ensure_ascii=False, sort_keys=True)
        full, full_metrics = memory_message(items, args.base_url, max_tokens=32768, compact_references=True)
        if full_metrics['injected_count'] != len(items):
            raise RuntimeError('Full reference rendering omitted source data')
        expected = {row['ref']: row for row in payload(full)}
        old, a = memory_message(items, args.base_url, max_tokens=768, compact_references=True)
        new, b = memory_message(items, args.base_url, max_tokens=768, compact_references=True, evidence_format='flat_v2')
        conserved = all(canonical(row, b) == expected[row['ref']] for row in payload(new))
        maps_conserved = all(value == full_metrics['evidence_references'][key] for key, value in b['evidence_references'].items())
        if not conserved or not maps_conserved or json.dumps(items, ensure_ascii=False, sort_keys=True) != before:
            raise RuntimeError('Evidence/source data changed during representation')
        old_ids, new_ids = set(a['injected_memory_ids']), set(b['injected_memory_ids'])
        rows.append({'id': case['id'], 'raw_sha256': sha256(raw_path),
            'old_tokens': a['memory_tokens'], 'new_tokens': b['memory_tokens'],
            'old_ids': a['injected_memory_ids'], 'new_ids': b['injected_memory_ids'],
            'gained_ids': sorted(new_ids - old_ids), 'lost_ids': sorted(old_ids - new_ids),
            'lossless_rows': conserved, 'complete_reference_maps': maps_conserved,
            'canonical_render_sha256': hashlib.sha256(json.dumps(expected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
    checks = {'all_82_cases': len(rows) == 82, 'all_data_conserved': all(r['lossless_rows'] and r['complete_reference_maps'] for r in rows),
        'all_within_budget': all(r['old_tokens'] <= 768 and r['new_tokens'] <= 768 for r in rows)}
    result = {'created_at': stamp(), 'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
        'source_report': str(args.source_report), 'source_report_sha256': sha256(args.source_report),
        'formatter_sha256': sha256(ROOT / 'app/context_memory.py'), 'source_sha256': sha256(Path(__file__)),
        'summary': {'cases': len(rows), 'cases_with_gains': sum(bool(r['gained_ids']) for r in rows),
            'cases_with_losses': sum(bool(r['lost_ids']) for r in rows),
            'old_total_injected': sum(len(r['old_ids']) for r in rows), 'new_total_injected': sum(len(r['new_ids']) for r in rows),
            'old_total_tokens': sum(r['old_tokens'] for r in rows), 'new_total_tokens': sum(r['new_tokens'] for r in rows)},
        'cases': rows,
        'scope': 'Existing development retrieval snapshots only;real llama.cpp tokenizer,768-token evidence limit.'
            ' Selected facts keep exact source text/date/role/spans and complete reference maps.'
            ' No fresh retrieval,model answers,reference-answer-driven selection,performance orsemantic-quality acceptance.'}
    write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'checks': checks, 'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8088/v1')
    main(parser.parse_args())
