"""Shared ownership lease for the running model processes."""

import fcntl
import os

from pathlib import Path


def hold_model_lease(directory=None):
    """Keep a shared lease across exec; the last model process exit releases it."""
    from project_paths import RUNTIME_DIR

    directory = Path(directory or RUNTIME_DIR / "runtime-data/edge-assistant/systemd")
    directory.mkdir(parents=True, exist_ok=True)
    lease = open(directory / "model-access.lock", "a+b")
    try:
        fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        lease.close()
        raise RuntimeError(
            "The model runtime is exclusively held by another operation"
        ) from None
    os.set_inheritable(lease.fileno(), True)
    return lease
