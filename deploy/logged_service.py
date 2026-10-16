"""Bound stdout/stderr files with stdlib logging; systemd owns supervision."""

import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.runtime import config, paths  # noqa: E402 — direct script entrypoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["llama", "asr", "speech", "app"])
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    directory = paths(config(args.profile))["data"] / "logs"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    handler = RotatingFileHandler(
        directory / (args.component + ".log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=4,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log = logging.getLogger("assistant-backend")
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    # systemd sends the stop signal to the whole cgroup, including the real
    # service child. Keep consuming its final logs until it exits; the native
    # systemd stop timeout handles a child that will not finish.
    signal.signal(signal.SIGINT, lambda *_: None)
    signal.signal(signal.SIGTERM, lambda *_: None)
    child = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "deploy/runtime.py"),
            args.component,
            "--profile",
            str(args.profile),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.info("backend_child_started component=%s pid=%s", args.component, child.pid)
    try:
        while data := child.stdout.readline(65536):
            log.info("%s", data.decode("utf-8", errors="replace").rstrip("\n"))
        result = child.wait()
        log.info(
            "backend_child_exited component=%s returncode=%s", args.component, result
        )
        return result if result >= 0 else 128 - result
    finally:
        child.stdout.close()
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
