"""Prepare and validate the locked long-memory embedding profile."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from app.memory_profiles import WINDOWS,model_path
from tools.quantize_bge_probe import main as export


def main():
    export()
    path=model_path(WINDOWS)
    print(json.dumps({'status':'READY','embedding_profile':WINDOWS,'model_path':str(path.resolve()),
        'scope':'Embedding artifacts verified; existing memory stores require explicit reindex.'},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
