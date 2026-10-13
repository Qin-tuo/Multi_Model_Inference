"""Explicit migration of an existing application memory directory."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import CACHE,DATA
from deploy.memory.migrate import abort,migrate,production_profile
from deploy.memory.runtime import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--abort',action='store_true',help='Remove an owned incomplete target; completed migration cannot be rolled back')
    args=parser.parse_args()
    directory=args.data_dir.expanduser().resolve()
    memory=directory/'memory'
    records=directory/'records.sqlite'
    try:
        service=load(DATA/'qdrant')
        if args.abort:result=abort(memory,service,records=records if records.exists() else None)
        else:
            profile=production_profile(CACHE/'models/bge-small-zh-v1.5-onnx')
            result=migrate(memory,service,profile,records=records if records.exists() else None)
    except Exception as error:
        print(json.dumps({'status':'FAIL','error':str(error),'action':'Fix the cause and repeat the same migrate-memory command; incomplete transitions block application access'},ensure_ascii=False),file=sys.stderr)
        return 1
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
