"""Exercise dated source recall through public interact using real local services."""
from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory_runtime import open_memory
from deploy.memory.runtime import load
from tools.project import DATA,stamp,write_json


def main():
    service=load(DATA/'qdrant');checks={};cases=[]
    with tempfile.TemporaryDirectory(prefix='source-memory-console-') as tmp:
        path=Path(tmp);data=path/'data';binding={};owner='synthetic-source-owner'
        def ask(label,lines,user=owner):
            script=path/(label+'.txt');script.write_text('\n'.join([*lines,'/quit'])+'\n')
            result=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'interact','--no-vision',
                '--user',user,'--data-dir',str(data),'--input-file',str(script)],
                capture_output=True,text=True,timeout=150,cwd=ROOT)
            answers=[line.removeprefix('助手：') for line in result.stdout.splitlines() if line.startswith('助手：')]
            cases.append({'case':label,'returncode':result.returncode,'answers':answers})
            checks[label+'_completed']=result.returncode==0 and len(answers)==len([line for line in lines if not line.startswith('/')])
            if result.returncode:cases[-1]['error_tail']=result.stderr[-1500:]
            return answers
        try:
            before=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
            ask('public_save',['请记住，我喜欢喝桂花茶。'])
            after=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
            binding=json.loads((data/'memory/store.json').read_text())
            with open_memory(data/'memory') as memory:
                saved=memory.list_memories(owner,limit=20)
                sources=[s for item in saved for s in item['sources']]
                checks['public_save_has_local_day_and_user_source']=len(saved)==1 and len(sources)==1 and sources[0].get('date') in {before,after} and sources[0].get('speaker')=='user'
                day=sources[0].get('date',before) if sources else before
                fixtures=[('wanted','2023-04-30','user',owner,'月映青竹，风送茶香'),
                    ('other-day','2023-05-01','user',owner,'雨打芭蕉，云遮旧月'),
                    ('assistant','2023-04-30','assistant',owner,'星落荒原，雪覆长桥'),
                    ('other-owner','2023-04-30','user','synthetic-other-owner','山含远黛，水映晴霞')]
                for key,date,speaker,user,poem in fixtures:
                    memory.remember(user,'event',key,'我分享的诗句是：'+poem+'。',source_id='synthetic-'+key,
                        request_id=key,source_date=date,source_speaker=speaker)
            saved_recall=ask('public_saved_date_recall',[day+'我说过我喜欢喝什么茶？'])
            checks['saved_date_recall_after_process_restart']=len(saved_recall)==1 and '桂花' in saved_recall[0]
            answers=ask('historical_sources',[
                '2023年4月30日，我分享了哪句诗？只回答诗句。','/new',
                '2023年4月30日，你分享了哪句诗？只回答诗句。','/new',
                '2023年4月29日，我分享了哪句诗？不知道就说不知道。'])
            checks['user_poem_excludes_wrong_day_role_and_owner']=len(answers)==3 and '月映青竹' in answers[0] and not any(s in answers[0] for s in ['雨打芭蕉','星落荒原','山含远黛'])
            checks['assistant_source_keeps_role']=len(answers)==3 and '星落荒原' in answers[1] and '月映青竹' not in answers[1]
            checks['missing_day_does_not_reuse_recent_answer']=len(answers)==3 and any(s in answers[2] for s in ['不知道','没有','不清楚','未找到']) and not any(s in answers[2] for s in ['月映青竹','星落荒原','雨打芭蕉','山含远黛'])
            guest=ask('empty_owner',['2023年4月30日，我分享了哪句诗？不知道就说不知道。'],user='synthetic-empty-owner')
            checks['empty_owner_does_not_receive_other_owner_sources']=len(guest)==1 and any(s in guest[0] for s in ['不知道','没有','不清楚','未找到']) and not any(s in guest[0] for s in ['月映青竹','星落荒原','雨打芭蕉','山含远黛'])
            with open_memory(data/'memory') as memory:
                checks['recall_did_not_add_memories']=len(memory.list_memories(owner,limit=20))==4 and memory.list_memories('synthetic-empty-owner',limit=20)==[]
        except Exception as error:
            checks['completed_without_error']=False;cases.append({'error_type':type(error).__name__,'message':str(error)})
        finally:
            marker=data/'memory/store.json'
            if marker.exists():binding=json.loads(marker.read_text())
            if binding.get('collection'):
                with closing(service.client()) as client:client.delete_collection(binding['collection'])
    report={'created_at':stamp(),'status':'PASS' if checks and all(checks.values()) else 'FAIL','checks':checks,'cases':cases,
        'scope':'Public interact with real Pipecat/llama/BGE/Mem0/authenticated Qdrant, explicit synthetic source fixtures and process/session restarts. No physical audio or independent memory-quality acceptance.'}
    write_json(ROOT/'benchmarks/results/memory-sources-console-smoke.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
