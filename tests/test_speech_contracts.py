# SPDX-License-Identifier: Apache-2.0
import io
import json
import wave
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import httpx
from app.audio import pcm_wav, wav_pcm
from app.config import Config
from app.stt import STT
from app.tts import LocalTTS
from app.pipeline import TextChunker, SpeechOutput, play_audio, PulseAudioAEC


def test_strict_audio_container_and_sample_format():
    raw = np.arange(100, dtype='<i2').tobytes()
    assert wav_pcm(pcm_wav(raw)) == raw
    for bad in [b'no wav', pcm_wav(raw)[:-4]]:
        with pytest.raises(ValueError): wav_pcm(bad)
    buf=io.BytesIO()
    with wave.open(buf,'wb') as w:
        w.setnchannels(2);w.setsampwidth(2);w.setframerate(16000);w.writeframes(raw)
    with pytest.raises(ValueError): wav_pcm(buf.getvalue())


def test_chinese_single_character_and_punctuation_are_not_dropped():
    chunker=TextChunker(first_chars=8,max_chars=12)
    chunks=[]
    for token in ['你','好。','请','停','下！']:
        chunks.extend(chunker.feed(token))
    chunks.extend(chunker.finish())
    assert chunks == ['你好。','请停下！']
    one=TextChunker();assert one.feed('停') == [] and one.finish() == ['停']
    long=TextChunker(4,8);chunks=long.feed('中文'*40)+long.finish()
    assert ''.join(chunks)=='中文'*40 and max(map(len,chunks))<=8


class FakeTTS:
    timeout=0.1
    def cancel(self,event): pass
    def synthesize(self,text,**kwargs):
        return {'audio':np.full(160,100,dtype=np.int16),'sample_rate':16000}


def test_file_output_receipt_and_retention_are_real(tmp_path):
    cfg=replace(Config.load().audio,output_directory=str(tmp_path),retained_files=1)
    for key in ['a'*32,'b'*32]:
        output=SpeechOutput(FakeTTS(),cfg,key)
        output.submit('你好。');output.submit('第二句。')
        receipt=output.close()
        assert receipt['kind']=='file_written' and receipt['bytes']==640
        with wave.open(receipt['path'],'rb') as w:assert w.getnframes()==320
    assert [p.name for p in tmp_path.glob('*.wav')]==['reply-'+('b'*32)+'.wav']
    assert not list(tmp_path.glob('*.partial'))


def test_synthesis_failure_does_not_publish_a_file(tmp_path):
    class Failed(FakeTTS):
        def synthesize(self,text,**kwargs):return {'audio':None,'error':'provider failed'}
    cfg=replace(Config.load().audio,output_directory=str(tmp_path))
    output=SpeechOutput(Failed(),cfg,'c'*32)
    output.submit('你好')
    with pytest.raises(RuntimeError,match='provider failed'):output.close()
    output.close(success=False)
    assert output.receipt is None and not list(tmp_path.iterdir())


def test_playback_failure_is_not_a_completion(monkeypatch):
    class Process:
        returncode=1
        def communicate(self,*args,**kwargs):return None,b'device gone'
        def poll(self):return 1
    monkeypatch.setattr('app.pipeline.subprocess.Popen',lambda *a,**k:Process())
    with pytest.raises(RuntimeError,match='device gone'):play_audio(np.ones(16,dtype=np.int16),16000,'chosen-sink')
    with pytest.raises(ValueError):play_audio(np.ones(16,dtype=np.int16),16000)


def test_asr_rejects_bad_array_without_submitting():
    asr=STT()
    calls=[]
    asr._client=httpx.Client(transport=httpx.MockTransport(lambda req:calls.append(req)))
    try:
        assert 'error' in asr.transcribe(np.ones((2,100),dtype=np.float32))
        assert 'error' in asr.transcribe(np.array([np.nan],dtype=np.float32))
        assert not calls
    finally:asr.unload()


@pytest.mark.parametrize('payload',[b'',b'abc'])
def test_tts_rejects_empty_or_partial_sample(payload):
    tts=LocalTTS()
    tts._client=httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(200,content=payload)))
    try:
        result=tts.synthesize('你好')
        assert result['audio'] is None and result['error']
    finally:tts.unload()


def test_failed_capability_check_closes_client(monkeypatch):
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(
        lambda req:httpx.Response(200,json={'asr':{'ready':True,'model_id':'wrong'}})),**kw))
    asr=STT();assert not asr.load() and asr._client is None


def test_aec_names_are_instance_scoped():
    first,second=PulseAudioAEC('mic'),PulseAudioAEC('mic')
    assert first.source_name!=second.source_name and first.sink_name!=second.sink_name


def test_microphone_mute_discards_old_epoch_without_device_operations():
    from app.pipeline import MicRecorder, AudioChunk
    from rich.console import Console
    mic=MicRecorder(Console())
    epoch=mic.input_epoch
    mic.audio_q.put(AudioChunk(b'x'*1024,epoch,0.0))
    mic.pause()
    assert mic.audio_q.empty() and not mic.listening.is_set() and mic.input_epoch!=epoch
    previous=mic.input_epoch
    mic.resume()
    assert mic.listening.is_set() and mic.input_epoch!=previous and mic.audio_q.empty()
    mic.stop()


def test_only_retained_managed_audio_can_be_downloaded(tmp_path):
    from app.turn_control import TurnController
    cfg=Config.load();cfg.audio.output_directory=str(tmp_path)
    control=TurnController(cfg,None,None)
    key='f'*32;path=tmp_path/('reply-'+key+'.wav');path.write_bytes(pcm_wav(b'\0'*32))
    assert control.audio_file(key)==path
    assert control.audio_file('../outside') is None
    path.unlink();path.symlink_to('/etc/os-release')
    assert control.audio_file(key) is None


def test_file_vad_offsets_cover_the_actual_pcm_duration():
    from app.pipeline import segment_pcm
    class Speech:
        def reset(self):pass
        def __call__(self, raw):return 1.0
    cfg=replace(Config.load().vad,silence_duration_ms=64,lookback_ms=0,min_utterance_secs=0.01)
    pcm=np.full(512*3,1000,dtype=np.int16).tobytes()
    segments=list(segment_pcm(pcm,cfg,Speech()))
    assert len(segments)==1
    assert segments[0].start_time==pytest.approx(0)
    assert segments[0].end_time-segments[0].start_time==pytest.approx(segments[0].duration)
