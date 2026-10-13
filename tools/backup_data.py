"""Create or verify a complete application data backup; never restores implicitly."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.data.backup import create,validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    backup=sub.add_parser('create');backup.add_argument('--data-dir',type=Path,required=True);backup.add_argument('--output',type=Path,required=True)
    verify=sub.add_parser('verify');verify.add_argument('--backup',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=create(args.data_dir,args.output) if args.action=='create' else validate(args.backup)
        print(json.dumps({'status':result['status'],'backup_id':result['backup_id'],'files':list(result['files']),
            'created_at':result['created_at'],'collections':{name:{'points':value['points'],'sha256':value['sha256']} for name,value in result['collections'].items()},
            'scope':'Quiescent Qdrant/SQLite application backup; restoration additionally requires current deletion and reminder delivery state'},ensure_ascii=False,indent=2))
        return 0
    except Exception as error:
        print(json.dumps({'status':'FAIL','error':str(error)},ensure_ascii=False),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
