"""Optional diagnostic timing around the real application and native frames.

Records no audio bytes, images, prompts or transcript text. Frame times are
mapped from each native pipeline clock, not from delayed observer delivery.
Wall spans include awaits/locks; they are not GPU or exclusive CPU durations.
"""
import functools
import threading
import time

from pipecat.frames.frames import MetricsFrame
from pipecat.observers.base_observer import BaseObserver


class StageTrace:
    def __init__(self,limit=20000):
        self.limit=limit;self.events=[];self.dropped=0;self._lock=threading.Lock();self._restore=[]

    def add(self,event):
        with self._lock:
            if len(self.events)>=self.limit:self.dropped+=1
            else:self.events.append(event)

    def sync(self,owner,name,label):
        original=getattr(owner,name)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            start=time.monotonic_ns();error=None
            try:return original(*args,**kwargs)
            except BaseException as failure:error=type(failure).__name__;raise
            finally:self.add({'kind':'span','name':label,'begin_ns':start,'end_ns':time.monotonic_ns(),
                'thread_id':threading.get_ident(),'error':error})
        setattr(owner,name,wrapped);self._restore.append((owner,name,original))

    def async_call(self,owner,name,label):
        original=getattr(owner,name)
        @functools.wraps(original)
        async def wrapped(*args,**kwargs):
            start=time.monotonic_ns();error=None
            try:return await original(*args,**kwargs)
            except BaseException as failure:error=type(failure).__name__;raise
            finally:self.add({'kind':'span','name':label,'begin_ns':start,'end_ns':time.monotonic_ns(),
                'thread_id':threading.get_ident(),'error':error})
        setattr(owner,name,wrapped);self._restore.append((owner,name,original))

    def attach(self,voice,memory):
        for name in ('search','remember','generation'):self.sync(memory,name,'memory.'+name)
        for name in ('_messages','_image_message'):self.async_call(voice.session,name,'session.'+name)
        self.async_call(voice.session.budget,'count','native_prompt_count')
        self.async_call(voice.speech,'deliver','speech.deliver_to_receipt')
        voice.worker.add_observer(NativeStageObserver(self,'voice'))
        voice.session.bot.worker.add_observer(NativeStageObserver(self,'model'))

    def detach(self):
        for owner,name,original in reversed(self._restore):setattr(owner,name,original)
        self._restore.clear()

    def report(self):
        with self._lock:return {'events':sorted(self.events,key=lambda e:e.get('begin_ns',e.get('monotonic_ns',0))),
            'dropped':self.dropped,'limit':self.limit,
            'scope':'Opt-in diagnostic instrumentation; wall spans include nested awaits/locks,not exclusive CPU or GPU time. Native metric semantics retained. No raw content. Observer overhead has not been qualified against an uninstrumented equivalent run.'}


class NativeStageObserver(BaseObserver):
    EVENTS={'VADUserStartedSpeakingFrame','VADUserStoppedSpeakingFrame','UserStartedSpeakingFrame','UserStoppedSpeakingFrame',
        'TranscriptionFrame','LLMFullResponseStartFrame','LLMFullResponseEndFrame','TTSStartedFrame','TTSStoppedFrame',
        'TTSSpeakFrame','UserImageRequestFrame','UserImageRawFrame','InterruptionFrame','ErrorFrame'}

    def __init__(self,trace,pipeline):
        super().__init__(observe_every_push=False)
        self.trace=trace;self.pipeline=pipeline;self.first_audio=set();self.first_text=set()

    async def on_push_frame(self,data):
        name=type(data.frame).__name__;source=data.source.name
        if name=='TTSStartedFrame':self.first_audio.discard(source)
        if name=='LLMFullResponseStartFrame':self.first_text.discard(source)
        if name=='TTSAudioRawFrame':
            if source in self.first_audio:return
            self.first_audio.add(source)
        elif name=='LLMTextFrame':
            if source in self.first_text:return
            self.first_text.add(source)
        elif name not in self.EVENTS and not isinstance(data.frame,MetricsFrame):return
        before=time.monotonic_ns();elapsed=data.source.get_clock().get_time();after=time.monotonic_ns()
        if elapsed<=0:
            self.trace.add({'kind':'unmapped_frame','pipeline':self.pipeline,'name':name,'frame_id':data.frame.id})
            return
        origin=(before+after)//2-elapsed
        row={'kind':'native_frame','pipeline':self.pipeline,'name':name,'processor':source,'frame_id':data.frame.id,
            'pipeline_timestamp_ns':data.timestamp,'monotonic_ns':origin+data.timestamp,
            'clock_mapping_uncertainty_ns':(after-before)//2,'observer_delivery_delay_ns':after-(origin+data.timestamp)}
        if isinstance(data.frame,MetricsFrame):row['metrics']=[{'type':type(metric).__name__,**metric.model_dump(mode='json')} for metric in data.frame.data]
        self.trace.add(row)
