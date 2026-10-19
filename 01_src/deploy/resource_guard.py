"""Shared runtime / exclusive heavy-build admission on this Jetson."""

import fcntl
import os

from pathlib import Path


def hold_model_lease(cache=None):
    """Keep a shared lease across exec; the last model process exit releases it."""
    from project_paths import RUNTIME_DIR

    cache = Path(cache or RUNTIME_DIR / "cache")
    cache.mkdir(parents=True, exist_ok=True)
    lease = open(cache / "ort-build.owner.lock", "a+b")
    try:
        fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        lease.close()
        raise RuntimeError(
            "A heavy model build owns the deployment lease; start model services after it finishes"
        ) from None
    os.set_inheritable(lease.fileno(), True)
    return lease
