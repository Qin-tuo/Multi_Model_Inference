"""Check native graph tracing with installed collectors before restarting models."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.project import sha256, stamp, write_json


def main(args):
    if args.run_dir.exists() or args.output.exists():
        raise RuntimeError('Preserve earlier evidence; use new paths')
    args.run_dir.mkdir(parents=True,mode=0o700)
    source = ROOT / 'benchmarks/fixtures/cuda_graph_update.cu'
    (args.run_dir / source.name).write_bytes(source.read_bytes())
    executable = args.run_dir / 'graph-update'
    command = ['/usr/local/cuda-13.2/bin/nvcc','-arch=sm_87','-O2',str(source),'-o',str(executable)]
    subprocess.run(command,check=True,timeout=60)
    report = {'created_at':stamp(),'source_sha256':sha256(source),'build_command':command,
        'scope':'12 synthetic one-kernel capture/update/launch cycles; collector compatibility only, not full model execution or performance.',
        'cases':[]}
    for version,mode in [('2026.3.1','node'),('2026.3.1','graph'),('2025.6.3','node')]:
        nsys = Path('/opt/nvidia/nsight-systems')/version/'target-linux-sbsa-armv8/nsys'
        label = f'{version}-{mode}'
        prefix = args.run_dir/label
        command = [str(nsys),'profile','--trace=cuda','--sample=none','--cpuctxsw=none',
            '--cuda-graph-trace='+mode,'--export=sqlite','--output='+str(prefix),str(executable)]
        with (args.run_dir/(label+'.log')).open('w') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        row = {'version':version,'granularity':mode,'command':command,'exit_code':result.returncode}
        with sqlite3.connect(f'file:{prefix}.sqlite?mode=ro',uri=True) as connection:
            tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
            row['activity_counts'] = {name:connection.execute('select count(*) from '+name).fetchone()[0]
                for name in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_GRAPH_TRACE') if name in tables}
            row['diagnostics'] = dict(Counter(r[0] for r in connection.execute('select text from DIAGNOSTIC_EVENT where severity in (2,3)')))
        report['cases'].append(row)
        write_json(args.output,report)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    report['completed_at'] = stamp()
    report['artifacts'] = [{'path':str(p.resolve()),'sha256':sha256(p),'bytes':p.stat().st_size}
        for p in sorted(args.run_dir.iterdir()) if p.is_file()]
    write_json(args.output,report)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
