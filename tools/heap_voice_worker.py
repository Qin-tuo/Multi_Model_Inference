"""Owned file-benchmark process with an explicitly reset bootstrap GC heap."""
import gc
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
index=sys.argv.index('--startup-heap') if '--startup-heap' in sys.argv else None
if index is None or index+1>=len(sys.argv) or sys.argv[index+1] not in {'collect','freeze'}:
    raise RuntimeError('This dedicated worker requires an explicit startup heap experiment')
if not gc.isenabled():raise RuntimeError('Automatic GC must remain enabled')
bootstrap={'owned_process':True,'frozen_before_reset':gc.get_freeze_count(),'thresholds':list(gc.get_threshold())}
# Reset only this fresh benchmark process,not normal application entry points.
gc.unfreeze();bootstrap['frozen_after_reset']=gc.get_freeze_count()
gc.collect();bootstrap['permanent_baseline_after_collect']=gc.get_freeze_count()
sys.argv[0]=str(ROOT/'tools/memory_voice_load.py')
runpy.run_path(sys.argv[0],run_name='__main__',init_globals={'GC_BOOTSTRAP_STATE':bootstrap})
