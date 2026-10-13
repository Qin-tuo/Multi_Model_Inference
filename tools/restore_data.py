"""Restore a backup with current control records into a separate application directory."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.data.restore import abort,restore
from deploy.memory.migrate import production_profile
from deploy.memory.runtime import load
from tools.project import DATA,CACHE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup',type=Path)
    parser.add_argument('--current-data-dir',type=Path)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--abort',action='store_true')
    args=parser.parse_args()
    try:
        if args.abort:result=abort(args.data_dir)
        else:
            if not args.backup or not args.current_data_dir:raise ValueError('Restoration requires --backup and --current-data-dir')
            from app.memory_profiles import profile_of,model_path
            selected=profile_of(json.loads((args.current_data_dir/'memory/store.json').read_text()))
            result=restore(args.backup,args.current_data_dir,args.data_dir,load(DATA/'qdrant'),
                production_profile(model_path(selected),selected))
        print(json.dumps(dict(result,data_directory=str(args.data_dir.resolve())),ensure_ascii=False,indent=2));return 0
    except Exception as error:
        print(json.dumps({'status':'FAIL','error':str(error),'action':'Keep both directories and repeat the same restore-data command after resolving the cause; do not remove transition files'},ensure_ascii=False),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
