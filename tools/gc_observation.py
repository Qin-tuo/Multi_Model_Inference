"""Passive bounded CPython GC timing; never changes collection policy."""
import gc
import threading
import time


class GCObservation:
    def __init__(self,limit=2000):
        self.limit=limit;self.events=[];self.active={};self.dropped=0

    def callback(self,phase,info):
        key=(threading.get_ident(),info['generation']);now=time.monotonic_ns()
        if phase=='start':self.active[key]=now
        elif phase=='stop':
            start=self.active.pop(key,None)
            if start is None:return
            if len(self.events)>=self.limit:self.dropped+=1;return
            self.events.append({'begin_ns':start,'end_ns':now,'elapsed_ms':(now-start)/1e6,
                'thread_id':key[0],'generation':key[1],'collected':info['collected'],'uncollectable':info['uncollectable']})

    def start(self):gc.callbacks.append(self.callback)
    def stop(self):
        if self.callback in gc.callbacks:gc.callbacks.remove(self.callback)
    def report(self):return {'events':self.events,'dropped':self.dropped,'thresholds':list(gc.get_threshold()),
        'enabled':gc.isenabled(),'scope':'Passive CPython GC callback wall time;no policy change. Timing correlation alone is not complete causality or observer-overhead qualification.'}
