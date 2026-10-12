"""Shared runtime / exclusive heavy-build admission on this Jetson."""
import fcntl
import os

from tools.project import CACHE


def hold_model_lease():
    """Keep a shared lease across exec; the last model process exit releases it."""
    CACHE.mkdir(parents=True, exist_ok=True)
    lease = open(CACHE / "ort-build.owner.lock", "a+b")
    try:
        fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        lease.close()
        raise RuntimeError("Heavy ORT build owns the deployment lease; start model services after it finishes") from None
    os.set_inheritable(lease.fileno(), True)
    return lease
