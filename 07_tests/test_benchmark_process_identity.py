"""A live auxiliary child must not be mistaken for a model service restart."""

import os
import socket
import subprocess
import sys

from app.config import Config
from deploy import services
from probe_coexistence import backend_identity


def test_real_listener_identity_survives_auxiliary_child_churn(monkeypatch):
    cfg = Config.load()
    pids = [os.getpid()]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        cfg.tts.base_url = cfg.stt.base_url = (
            f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
        )
        monkeypatch.setattr(
            services,
            "properties",
            lambda name: {
                "main_pid": os.getpid(),
                "NRestarts": "0",
                "cgroup_pids": list(pids),
            },
        )
        before = backend_identity("speech", cfg)
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        try:
            pids.append(child.pid)
            after = backend_identity("speech", cfg)
            assert before["observed_cgroup_pids"] != after["observed_cgroup_pids"]
            assert before["identity"] == after["identity"]
            assert child.pid not in after["stable_pids"]
        finally:
            child.terminate()
            child.wait(5)
