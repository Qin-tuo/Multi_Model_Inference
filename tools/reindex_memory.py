"""Upgrade a native memory directory to the bound long-document profile."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory_profiles import WINDOWS,model_path
from deploy.memory.migrate import production_profile
from deploy.memory.reindex import reindex,abort


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--abort',action='store_true');args=parser.parse_args()
    try:
        result=abort(args.data_dir) if args.abort else reindex(args.data_dir,production_profile(model_path(WINDOWS),WINDOWS))
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except Exception as error:
        print(json.dumps({'status':'FAIL','error':str(error),'action':'Resolve the cause and resume reindex-memory; incomplete transitions block memory access.'},ensure_ascii=False),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
