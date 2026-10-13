"""Public file-voice entry using the managed store; no physical audio acceptance."""
from contextlib import closing
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import wave

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.project import DATA,stamp,write_json,sha256
from deploy.memory.runtime import load
from app.memory_profiles import profile_of,entity_metadata
from app.memory_store import collection_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/managed-memory-voice-smoke.json')
    args=parser.parse_args()
    prior=json.loads((ROOT/'benchmarks/results/voice-assistant-smoke.json').read_text())
    inputs=[Path(prior['artifacts'])/f'input-{i}.wav' for i in (0,1)]
    if not all(p.is_file() for p in inputs):raise RuntimeError('Prepare the existing synthetic voice-assistant fixtures first')
    artifacts=DATA/'fixtures'/('managed-memory-voice-'+uuid.uuid4().hex);artifacts.mkdir(parents=True)
    service=load(DATA/'qdrant')
    with tempfile.TemporaryDirectory(prefix='managed-voice-state-') as tmp:
        state=Path(tmp)
        result=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'voice','--no-vision',
            '--user','synthetic-native-voice','--data-dir',str(state),'--input-wav',*map(str,inputs),
            '--output-wav',str(artifacts/'answers.wav'),'--report',str(artifacts/'turns.json')],
            capture_output=True,text=True,timeout=150,cwd=ROOT)
        detail=json.loads((artifacts/'turns.json').read_text()) if (artifacts/'turns.json').exists() else {}
        binding=json.loads((state/'memory/store.json').read_text()) if (state/'memory/store.json').exists() else {}
        turns=detail.get('turns',[])
        checks={'public_voice_completed':result.returncode==0,'native_bound':binding.get('backend')=='qdrant_server',
            'two_complete_turns':len(turns)==2 and all(t.get('speech_receipt',{}).get('completion')=='wav_file_flushed' for t in turns),
            'recalled_fact_in_reply':len(turns)==2 and '茉莉' in turns[1].get('response',''),
            'no_pipeline_errors':not detail.get('errors',[])}
        if binding.get('collection'):
            with closing(service.client()) as client:
                checks['fact_actually_persisted']=client.count(binding['collection'],exact=True).count==1
                for name,expected in [(binding['collection']+'_entities',entity_metadata(binding)),(binding['collection'],collection_metadata(binding))]:
                    if client.collection_exists(name):
                        if client.get_collection(name).config.metadata!=expected:raise RuntimeError('Refusing to clean an unrelated voice fixture collection')
                        client.delete_collection(name)
        audio={}
        if (artifacts/'answers.wav').is_file():
            with wave.open(str(artifacts/'answers.wav')) as source:
                audio={'frames':source.getnframes(),'rate':source.getframerate(),'channels':source.getnchannels()}
        checks['audio_file_nonempty']=audio.get('frames',0)>16000
        report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'embedding_profile':profile_of(binding) if binding else None,
            'scope':'Two existing synthetic WAV utterances through public voice command with real ASR/BGE/llama/TTS, authenticated bound Mem0/Qdrant store and WAV output. No physical audio or full quality/latency acceptance.',
            'input_sha256':[sha256(p) for p in inputs],'turns':turns,'audio':audio,'artifacts':str(artifacts),'returncode':result.returncode}
        if result.returncode:report['error_tail']=result.stderr[-1800:]
        write_json(args.output,report)
        print(json.dumps({'status':report['status'],'checks':checks},ensure_ascii=False))
        return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
