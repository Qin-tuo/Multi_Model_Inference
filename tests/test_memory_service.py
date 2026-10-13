"""Real authenticated Qdrant metadata binding; no embedding/model inference."""
from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing

from app.memory_store import StoreBinding
from deploy.memory.prepare import BINARY
from deploy.memory.runtime import prepare,load,health,private_key


class MemoryCredentialTests(unittest.TestCase):
    def test_private_credential_and_local_edits_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=prepare(Path(tmp)/'service',port=19333)
            self.assertEqual(service,load(service.directory))
            self.assertEqual(service.key_path.stat().st_mode&0o777,0o600)
            service.key_path.chmod(0o644)
            with self.assertRaises(RuntimeError):private_key(service.key_path)
            service.key_path.chmod(0o600)
            config=service.directory/'config.yaml'
            config.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'local edits'):prepare(service.directory,port=19333)
            self.assertEqual(config.read_text(),'{}')


@unittest.skipUnless(BINARY.exists(),'Pinned Qdrant binary must be prepared')
class NativeMemoryBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='memory-binding-')
        cls.root=Path(cls.temp.name)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        cls.service=prepare(cls.root/'service',port=port)
        cls.log=(cls.root/'service.log').open('w')
        cls.process=subprocess.Popen([sys.executable,'-c',
            'from deploy.memory.runtime import serve; import sys; serve(sys.argv[1])',str(cls.service.directory)],
            stdout=cls.log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if health(cls.service.directory)['ready']:return
            if cls.process.poll() is not None:break
            time.sleep(.1)
        cls.process.terminate();cls.process.wait(timeout=10);cls.log.close();cls.temp.cleanup()
        raise RuntimeError('Authenticated native fixture did not start')
    @classmethod
    def tearDownClass(cls):
        cls.process.terminate();cls.process.wait(timeout=10);cls.log.close();cls.temp.cleanup()
    def directory(self,name):
        path=self.root/name;path.mkdir();return path
    def bind(self,path):return StoreBinding(path,self.service,embedding_sha256='a'*64,dimensions=8)
    def test_authentication_binding_reopen_and_copy_rejection(self):
        self.assertTrue(health(self.service.directory)['authentication_required'])
        path=self.directory('original')
        binding=self.bind(path)
        try:
            binding.ready();collection=binding.collection
            with self.assertRaises(BlockingIOError):self.bind(path)
            copy=self.root/'copy';shutil.copytree(path,copy)
            with self.assertRaisesRegex(RuntimeError,'migration'):self.bind(copy)
        finally:binding.close()
        reopened=self.bind(path)
        try:self.assertEqual(reopened.collection,collection)
        finally:reopened.close()
    def test_missing_ready_collection_is_not_recreated(self):
        path=self.directory('missing');binding=self.bind(path)
        binding.ready();collection=binding.collection;binding.close()
        with closing(self.service.client()) as client:client.delete_collection(collection)
        with self.assertRaisesRegex(RuntimeError,'missing'):self.bind(path)
        with closing(self.service.client()) as client:self.assertFalse(client.collection_exists(collection))
    def test_wrong_collection_metadata_and_embedding_are_rejected(self):
        path=self.directory('metadata');binding=self.bind(path)
        binding.ready();collection=binding.collection;binding.close()
        with self.assertRaisesRegex(RuntimeError,'migration'):
            StoreBinding(path,self.service,embedding_sha256='b'*64,dimensions=8)
        with closing(self.service.client()) as client:client.update_collection(collection,metadata={'unrelated_store':True})
        with self.assertRaisesRegex(RuntimeError,'metadata'):self.bind(path)
    def test_legacy_embedded_directory_requires_migration(self):
        path=self.directory('legacy');(path/'qdrant').mkdir()
        with self.assertRaisesRegex(RuntimeError,'migration'):self.bind(path)
        self.assertFalse((path/'store.json').exists())
