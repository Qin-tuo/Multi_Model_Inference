"""Service argument, ownership and local-edit protection contracts."""
import os
import fcntl
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
import time
from unittest.mock import patch

from deploy import services


class ServiceConfigTests(unittest.TestCase):
    def test_probe_timeout_is_reported_without_claiming_backend_exit(self):
        with patch.object(services,'command',side_effect=subprocess.TimeoutExpired(['labelled_probe'],8)):
            result=services.probe('camera')
        self.assertFalse(result['ready'])
        self.assertEqual(result['reason'],'ROS_probe_timeout')

    def test_dependencies_and_rejected_component_names(self):
        self.assertEqual(services.selected("speech,pose"), ["asr", "speech", "camera", "pose"])
        with self.assertRaises(ValueError): services.selected("camera;arbitrary-command")

    def test_exec_arguments_are_literal_and_unit_newlines_are_rejected(self):
        self.assertEqual(services.quote('/tmp/a%/$X/"name"', executable=True), '"/tmp/a%%/$$X/\\"name\\""')
        with self.assertRaises(ValueError): services.quote("path\nExecStart=unexpected")

    def test_socket_ownership_requires_the_actual_process(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); listener.listen()
            with patch.dict(services.PORTS, {"llama": listener.getsockname()[1]}):
                self.assertTrue(services.owns_port("llama", [os.getpid()]))
                self.assertFalse(services.owns_port("llama", []))

    def test_unmanaged_unit_and_local_edits_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            units, links = Path(temporary) / "units", Path(temporary) / "links"
            units.mkdir(); links.mkdir()
            edited = units / services.unit("llama")
            edited.write_text("local edit to preserve\n")
            with patch.object(services, "UNIT_DIR", units), patch.object(services, "LINK_DIR", links):
                with self.assertRaisesRegex(RuntimeError, "local edits"):
                    services.install()
                self.assertEqual(edited.read_text(), "local edit to preserve\n")
                (links / services.unit("llama")).write_text("unrelated installed unit")
                with self.assertRaisesRegex(RuntimeError, "unrelated"):
                    services.install()
                self.assertEqual((links / services.unit("llama")).read_text(), "unrelated installed unit")

    def test_model_lease_survives_exec_and_releases_on_process_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = os.environ.copy(); env["ASSISTANT_CACHE_DIR"] = temporary
            code = "from deploy.resource_guard import hold_model_lease; import os; lease=hold_model_lease(); os.execv('/usr/bin/sleep',['sleep','3'])"
            child = subprocess.Popen([sys.executable, "-c", code], cwd=services.ROOT, env=env)
            try:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if (Path('/proc') / str(child.pid) / 'cmdline').read_bytes().startswith(b'sleep\0'): break
                    time.sleep(.01)
                else: self.fail("Fixture did not reach the exec'd process")
                with open(Path(temporary) / "ort-build.owner.lock", "a+b") as lock:
                    with self.assertRaises(BlockingIOError): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    child.terminate(); child.wait(timeout=3)
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if child.poll() is None: child.terminate(); child.wait(timeout=3)


if __name__ == "__main__": unittest.main()
