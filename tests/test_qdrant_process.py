"""Native process ownership/listener qualification, without model or user data."""
import socket
from pathlib import Path
import tempfile
import unittest

from deploy.memory.prepare import BINARY
from deploy.memory.server import QdrantProcess


@unittest.skipUnless(BINARY.exists(),'Pinned native Qdrant binary must be prepared')
class NativeQdrantProcessTests(unittest.TestCase):
    def test_loopback_readiness_exclusive_ownership_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            path=Path(tmp)/'owned'
            with QdrantProcess(path,port=port) as server:
                process=server.process
                self.assertIsNone(process.poll())
                with self.assertRaises((RuntimeError,BlockingIOError)):
                    QdrantProcess(path,port=port).start()
                self.assertIsNone(process.poll())
            self.assertIsNotNone(process.poll())
            with QdrantProcess(path,port=port) as restarted:
                self.assertNotEqual(process.pid,restarted.process.pid)
    def test_unknown_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'config.yaml';marker.write_text('unrelated configuration')
            with self.assertRaises(RuntimeError):QdrantProcess(tmp).start()
            self.assertEqual(marker.read_text(),'unrelated configuration')
