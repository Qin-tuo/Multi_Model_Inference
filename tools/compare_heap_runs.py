"""Compare startup GC experiments while keeping startup/cleanup costs visible."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import stamp,write_json


def main(args):
    if args.output.exists():raise RuntimeError('Do not overwrite a heap comparison')
    before=json.loads(args.baseline.read_text());after=json.loads(args.candidate.read_text())
    checks={'same_application':before['reader_fingerprint']==after['reader_fingerprint'],
        'same_memory_recipe':before['config']==after['config'],'same_retrieval':before['retrieval_config']==after['retrieval_config'],
        'same_routing':before['skip_literal_arithmetic']==after['skip_literal_arithmetic'],
        'same_interval':before['turn_interval_seconds']==after['turn_interval_seconds']==10,
        'same_inputs':[(t['label'],t.get('input_sha256')) for t in before['turns']]==[(t['label'],t.get('input_sha256')) for t in after['turns']],
        'twenty_attempts':len(before['turns'])==len(after['turns'])==20,
        'same_owned_bootstrap':before['gc_bootstrap_state']==after['gc_bootstrap_state'],
        'both_automatic_gc_enabled':before['gc_observation']['enabled'] and after['gc_observation']['enabled'],
        'same_gc_thresholds':before['gc_observation']['thresholds']==after['gc_observation']['thresholds'],
        'both_policies_restored':before['startup_heap_policy']['policy_restored'] and after['startup_heap_policy']['policy_restored']}
    seeds=[json.loads((Path(r['run_directory'])/'clone-receipt.json').read_text()) for r in (before,after)]
    checks['same_pristine_snapshot']=seeds[0]['backup_manifest_sha256']==seeds[1]['backup_manifest_sha256']
    rows=[]
    for r in (before,after):
        policy=r['startup_heap_policy'];events=[e for e in r['gc_observation']['events'] if r['workload_begin_ns']<=e['begin_ns']<=r['workload_end_ns']]
        rows.append({'mode':policy['mode'],'status':r['status'],'workload_completed':r['checks']['workload_completed'],
            'voice_latency':r['voice_latency'],'event_loop_lag':r['event_loop_lag'],
            'workload_gc_count':len(events),'workload_gc_by_generation':dict(Counter(e['generation'] for e in events)),
            'workload_gc_max_ms':max((e['elapsed_ms'] for e in events),default=0),
            'startup_collection_ms':(policy['startup_end_ns']-policy['startup_begin_ns'])/1e6,
            'closing_collection_ms':(policy['close_end_ns']-policy['close_begin_ns'])/1e6,
            'frozen_objects':policy['frozen_objects'],'restored_permanent_objects':policy['frozen_objects_after_close'],
            'query_p95_ms':r['query_summary']['p95_ms'],'write_p95_ms':r['write_summary']['p95_ms'],
            'resources_including_startup_and_readbacks':r['resources'],
            'bridge_epoch_resets_including_startup':r['bridge']['counts'].get('reset:ros_input_epoch',0)})
    report={'created_at':stamp(),'status':'DEVELOPMENT_COMPARISON' if all(checks.values()) and all(r['workload_completed'] for r in rows) else 'INCOMPLETE',
        'checks':checks,'variants':rows,'source_reports':[str(args.baseline),str(args.candidate)],
        'scope':'One short file-workload pair withlivecamera and realmodels. Workload GC excludes explicitly reported startup/close collections. A short collect-only run withoutgeneration2 cannot establish long-term elimination;no8h/memory-leak,acoustic,independentquality orfinalJT claim. Startupfreezing remains benchmark-only.'}
    write_json(args.output,report);print(json.dumps({'status':report['status'],'checks':checks,'variants':rows},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
