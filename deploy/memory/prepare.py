"""Fetch the pinned official CPU Qdrant binary and source into AI_Stack."""
import hashlib
from pathlib import Path
import subprocess
import tarfile
import urllib.request

VERSION='1.19.2'
COMMIT='016542aa5deb6c66380bb137badf73d54f742bde'
ARCHIVE='qdrant-aarch64-unknown-linux-musl.tar.gz'
SHA256='6970b93b56fa1203f0cea47d2f330fdb654988fd56617aa77e8478cffd3c0ccb'
STACK=Path.home()/'AI_Stack'
BINARY=STACK/'installed'/('qdrant-'+VERSION)/'qdrant'


def main():
    archive=STACK/'downloads'/('qdrant-'+VERSION+'-'+ARCHIVE)
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        temporary=archive.with_suffix('.part')
        with urllib.request.urlopen(f'https://github.com/qdrant/qdrant/releases/download/v{VERSION}/{ARCHIVE}',timeout=60) as src,temporary.open('wb') as dst:
            while chunk:=src.read(1024*1024):dst.write(chunk)
        temporary.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:raise RuntimeError('Official archive SHA256 mismatch')
    BINARY.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as f:
        member=next(m for m in f.getmembers() if m.name.lstrip('./')=='qdrant' and m.isfile())
        data=f.extractfile(member).read()
        if BINARY.exists() and BINARY.read_bytes()!=data:raise RuntimeError('Existing Qdrant binary differs; refusing to overwrite')
        if not BINARY.exists():BINARY.write_bytes(data);BINARY.chmod(0o755)
    source=STACK/'frameworks/qdrant'
    if not source.exists():subprocess.run(['git','clone','--depth','1','--branch','v'+VERSION,'https://github.com/qdrant/qdrant.git',str(source)],check=True)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=COMMIT:raise RuntimeError('Qdrant source checkout differs from the lock')
    subprocess.run([str(BINARY),'--version'],check=True)


if __name__=='__main__':main()
