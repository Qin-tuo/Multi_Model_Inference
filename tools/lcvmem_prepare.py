"""Pin official Chinese memory benchmark assets and an isolated scoring environment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
STACK=Path.home()/'AI_Stack'
CODE=STACK/'frameworks/CLongEval'
DATA=STACK/'datasets/CLongEval/long_conversation_memory-small.jsonl'
SCORER=STACK/'environments/memory-eval/bin/python'
COMMIT='81ba789bdbf83b88579c0a0ae780fa04090899cd'
REVISION='178d74d88f525f494441f94ef8cff9185a3d3b3a'
DATA_SHA='421fbf602154bcfcd55f943edf636ef8734183ffec68f4580de178b7ec3b5a25'
URL=f'https://huggingface.co/datasets/zexuanqiu22/CLongEval/resolve/{REVISION}/1-2_long_conversation_memory/small.jsonl'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    repo='https://github.com/zexuanqiu/CLongEval.git'
    if not CODE.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',repo,str(CODE)],check=True)
        subprocess.run(['git','-C',str(CODE),'sparse-checkout','set','--no-cone','/LICENSE','/README.md','/metrics.py','/eval.py','/post_processing_for_stacked_tasks.py','/config'],check=True)
        subprocess.run(['git','-C',str(CODE),'checkout','--detach',COMMIT],check=True)
    if subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()!=COMMIT:
        raise RuntimeError('Existing CLongEval checkout differs; local changes are preserved')
    subprocess.run(['git','-C',str(CODE),'diff','--exit-code',COMMIT,'--','metrics.py','post_processing_for_stacked_tasks.py'],check=True)
    DATA.parent.mkdir(parents=True,exist_ok=True)
    if not DATA.exists():
        temp=DATA.with_suffix('.download')
        with urllib.request.urlopen(URL,timeout=30) as stream,temp.open('wb') as out:
            while block:=stream.read(1024*1024):out.write(block)
        if sha(temp)!=DATA_SHA:raise RuntimeError('CLongEval data checksum differs')
        temp.replace(DATA)
    if sha(DATA)!=DATA_SHA:raise RuntimeError('Existing CLongEval data checksum differs')
    if not SCORER.exists():subprocess.run([sys.executable,'-m','venv',str(SCORER.parent.parent)],check=True)
    subprocess.run([str(SCORER),'-m','pip','install','--require-hashes','--no-deps','-r',str(ROOT/'deploy/evaluation/metrics.lock')],check=True)
    print(json.dumps({'dataset':str(DATA),'sha256':DATA_SHA,'upstream':str(CODE),'commit':COMMIT,'scoring_python':str(SCORER)},ensure_ascii=False))


if __name__=='__main__':main()
