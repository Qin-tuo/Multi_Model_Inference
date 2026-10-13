"""Exclusive opt-in startup GC experiment; not the application's default.

Automatic collection stays enabled. Freeze omits startup objects from later
cyclic scans, so this owner must unfreeze/collect after all voice work closes.
"""
import gc
import threading
import time

_OWNER=threading.Lock()


class StartupHeapPolicy:
    def __init__(self,mode,*,permanent_baseline):
        if mode not in {'collect','freeze'}:raise ValueError('Unknown startup heap experiment')
        if type(permanent_baseline) is not int or permanent_baseline<0:raise ValueError('Owned process baseline required')
        self.permanent_baseline=permanent_baseline
        self.mode=mode;self.owned=False;self.frozen=False;self.closed=False;self.evidence={'mode':mode}

    def start(self):
        if self.owned or self.closed:raise RuntimeError('Heap policy already used')
        if not _OWNER.acquire(blocking=False):raise RuntimeError('Another startup heap policy is active')
        self.owned=True
        try:
            if not gc.isenabled() or gc.get_freeze_count()!=self.permanent_baseline:raise RuntimeError('Experiment requires enabled GC and no foreign frozen heap beyond the owned bootstrap baseline')
            self.thresholds=gc.get_threshold();start=time.monotonic_ns()
            collected=gc.collect()
            if not gc.isenabled() or gc.get_freeze_count()!=self.permanent_baseline or gc.get_threshold()!=self.thresholds:
                raise RuntimeError('GC policy changed during startup collection')
            if self.mode=='freeze':gc.freeze();self.frozen=True
            self.evidence.update(startup_begin_ns=start,startup_end_ns=time.monotonic_ns(),startup_collected=collected,
                frozen_objects=gc.get_freeze_count(),thresholds=list(self.thresholds),automatic_gc_enabled=gc.isenabled())
            self.evidence['permanent_baseline']=self.permanent_baseline
        except BaseException:
            if self.frozen:gc.unfreeze();self.frozen=False;gc.collect()
            self.owned=False;_OWNER.release();raise

    def close(self):
        if not self.owned:return
        try:
            start=time.monotonic_ns()
            self.evidence['workload_frozen_objects_at_close']=gc.get_freeze_count()
            if self.frozen:gc.unfreeze();self.frozen=False
            collected=gc.collect()
            self.evidence.update(close_begin_ns=start,close_end_ns=time.monotonic_ns(),close_collected=collected,
                frozen_objects_after_close=gc.get_freeze_count(),automatic_gc_after_close=gc.isenabled(),
                thresholds_after_close=list(gc.get_threshold()))
            self.evidence['policy_restored']=gc.isenabled() and gc.get_freeze_count()==self.permanent_baseline and gc.get_threshold()==self.thresholds
        finally:self.owned=False;self.closed=True;_OWNER.release()

    def report(self):return {**self.evidence,'scope':'Startup-only collect versus collect+freeze experiment. Automatic GC and thresholds unchanged;new objects remain collectible. Frozen startup cycles can survive until ownerclose;unfreeze/cleanup checked. Not production lifecycle or8h qualification.'}
