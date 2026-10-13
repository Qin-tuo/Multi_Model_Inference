"""Public text entry using the managed native store, with isolated synthetic data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import DATA,stamp,write_json
from deploy.memory.runtime import load
from contextlib import closing


def main():
    service=load(DATA/'qdrant')
    with tempfile.TemporaryDirectory(prefix='native-memory-console-') as tmp:
        directory=Path(tmp);script=directory/'input.txt'
        script.write_text('请记住，我喜欢喝桂花茶。\n我喜欢喝什么茶？\n/new\n我喜欢喝什么茶？\n更正，我现在喜欢喝茉莉花茶。\n/new\n我喜欢喝什么茶？\n请忘记我喜欢喝茉莉花茶这件事。\n/memories\n/quit\n')
        result=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'interact','--no-vision',
            '--user','synthetic-native-owner','--data-dir',str(directory/'data'),'--input-file',str(script)],
            capture_output=True,text=True,timeout=120,cwd=ROOT)
        binding_path=directory/'data/memory/store.json'
        binding=json.loads(binding_path.read_text()) if binding_path.exists() else {}
        answers=[line.removeprefix('助手：') for line in result.stdout.splitlines() if line.startswith('助手：')]
        checks={'public_cli_completed':result.returncode==0,'native_bound_store':binding.get('backend')=='qdrant_server' and binding.get('status')=='READY',
            'recall_before_and_after_new_session':len(answers)==6 and all('桂花' in answers[i] for i in [1,2]),
            'corrected_recall_after_new_session':len(answers)==6 and '茉莉' in answers[4] and '桂花' not in answers[4],
            'explicit_forget_confirmed':len(answers)==6 and any(w in answers[5] for w in ['删除','忘记','移除']),
            'empty_after_forget':'暂无记录。' in result.stdout}
        with closing(service.client()) as client:
            if binding.get('collection'):
                info=client.get_collection(binding['collection'])
                checks['native_collection_metadata_matches']=info.config.metadata['assistant_store_id']==binding['store_id']
                checks['actual_native_store_empty']=client.count(binding['collection'],exact=True).count==0
                client.delete_collection(binding['collection'])
        report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'answers':answers,
            'returncode':result.returncode,'scope':'Public interact command with actual Pipecat/llama/BGE/Mem0/authenticated native Qdrant; synthetic temporary profile, save/recall/session reset/correction/forget. Not independent memory quality or physical audio acceptance.'}
        if result.returncode:report['error_tail']=result.stderr[-1800:]
        write_json(ROOT/'benchmarks/results/managed-memory-console-smoke.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
