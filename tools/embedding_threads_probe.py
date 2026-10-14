"""Screen CPU ORT thread counts through the real Mem0 BGE window provider.

Use a fresh subprocess for each policy. Record-wise int8 batches,weights,
windowing and normalization are identical to the application. No store writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['MEM0_TELEMETRY'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from tools.project import stamp, write_json, sha256


def main(args):
    if args.output.exists() or args.run_dir.exists():
        raise RuntimeError('Use fresh artifact and report paths')
    import numpy as np
    import onnxruntime as ort
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.huggingface import HuggingFaceEmbedding
    from app.memory import build_mem0_config
    from app.adapters.embedding_windows import BGEWindowAdapter
    from app.memory_profiles import HYBRID, model_path
    from tools.managed_memory_capacity import workload
    from tools.memory_capacity import MemoryMonitor, backend_state, statistics
    path = model_path(HYBRID)
    corpus, recipe, prefix, _ = workload(10000, personal=True, profile=HYBRID)
    cases = []
    for index in range(20):
        short = f'临时测量T{index:03d}要求离开前关闭台灯。'
        for kind, text, action in [('long', prefix + short, 'add'), ('short', short, 'add'),
                ('query', corpus['query_template'].format(index=index * 500), 'search')]:
            cases.append({'id': f'{kind}-{index}', 'kind': kind, 'text': text, 'action': action})
    args.run_dir.mkdir(parents=True, mode=0o700)
    write_json(args.run_dir / 'inputs.json', cases)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    config = build_mem0_config(args.run_dir / 'unused-store', path, embedding_backend='onnx')['embedder']['config']
    config['model_kwargs']['model_kwargs']['session_options'] = options
    before = backend_state()
    encoder = BGEWindowAdapter(HuggingFaceEmbedding(BaseEmbedderConfig(**config)))
    session = encoder.model[0].auto_model.model
    actual = session.get_session_options()
    if actual.intra_op_num_threads != args.threads or session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('Actual ORT session differs from requested CPU policy')
    # Warm the three shapes explicitly; startup/loading time is not throughput.
    warm_started = time.monotonic_ns()
    for case in cases[:3]:
        encoder.embed(case['text'], memory_action=case['action'])
    warm_ms = (time.monotonic_ns() - warm_started) / 1e6
    rows = []
    arrays = {}
    with MemoryMonitor() as monitor:
        for case in cases:
            encoder._query = None  # Force actual query encoding,not last-query reuse.
            start = time.monotonic_ns()
            value = encoder.embed(case['text'], memory_action=case['action'])
            elapsed = (time.monotonic_ns() - start) / 1e6
            vectors = np.asarray(value, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != 512 or not np.isfinite(vectors).all():
                raise RuntimeError('Invalid actual BGE vectors')
            arrays[case['id']] = vectors
            rows.append({'id': case['id'], 'kind': case['kind'], 'elapsed_ms': elapsed,
                'shape': list(vectors.shape), 'input_sha256': hashlib.sha256(case['text'].encode()).hexdigest(),
                'vectors_sha256': hashlib.sha256(vectors.tobytes()).hexdigest()})
    np.savez(args.run_dir / 'vectors.npz', **arrays)
    after = backend_state()
    report = {'created_at': stamp(), 'status': 'MEASURED', 'threads': args.threads,
        'inter_op_threads': actual.inter_op_num_threads, 'providers': session.get_providers(),
        'source_sha256': sha256(Path(__file__)), 'model_sha256': sha256(path / 'onnx/model.onnx'),
        'input_sha256': sha256(args.run_dir / 'inputs.json'), 'recipe': recipe,
        'warmup_ms': warm_ms, 'rows': rows, 'artifacts': str(args.run_dir.resolve()),
        'summary': {kind: statistics([row for row in rows if row['kind'] == kind]) for kind in ('long', 'short', 'query')},
        'resources': monitor.report(), 'backends_before': before, 'backends_after': after,
        'backend_identities_unchanged': before == after,
        'scope': 'Real Mem0/SentenceTransformers/CPU ORT window encoding only;20 synthetic inputs perkind.'
            ' No storage,query ranking,foreground speech or continuously measured visual rate.'
            ' Fixed record-wise batches retained;vectors saved for exact cross-policy verification.'
            ' Screening only,not full-load or quality acceptance. Production settings unchanged.'}
    write_json(args.output, report)
    print(json.dumps({'threads': args.threads, 'summary': report['summary'],
        'backend_identities_unchanged': report['backend_identities_unchanged']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1, 2, 4], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
