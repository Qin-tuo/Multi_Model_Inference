# SPDX-License-Identifier: Apache-2.0
"""Owned-service coordination and native process identity for build/validation tools."""

from contextlib import contextmanager
import fcntl
import psutil
from deploy import runtime, services


@contextmanager
def exclusive_build(cfg, output, *, allow_failed_stop=False):
    """Keep deployment operations and model startup excluded until restoration."""
    with services.operation(cfg):
        before = services.state(list(services.ORDER), cfg)
        runtime.write_json(output / "services-before.json", before)
        for name, row in before["services"].items():
            if row.get("cgroup_pids") and not row.get("managed_by_project"):
                raise RuntimeError("Build will not stop an unrelated service: " + name)
        app = before["services"].get("app", {})
        if app.get("health", {}).get("phase") not in (None, "idle"):
            raise RuntimeError("Wait for the current application turn before building")
        active = [
            n
            for n in services.ORDER
            if before["services"][n].get("ActiveState") == "active"
        ]
        lease = None
        try:
            stopped = services.stop(list(services.ORDER), cfg)
            runtime.write_json(output / "services-stopped.json", stopped)
            if stopped["status"] != "STOPPED":
                rows = stopped.get("services", {})
                terminal_and_empty = set(rows) == set(services.ORDER) and all(
                    row.get("managed_by_project") is True
                    and row.get("cgroup_pids") == []
                    and row.get("main_pid") == 0
                    and row.get("ActiveState") in ("inactive", "failed")
                    for row in rows.values()
                )
                if not (allow_failed_stop and terminal_and_empty):
                    raise RuntimeError(
                        "Service shutdown was not clean; see services-stopped.json"
                        if terminal_and_empty
                        else "Managed processes have not stopped"
                    )
                runtime.write_json(
                    output / "shutdown-warning.json",
                    {
                        "status": "UNCLEAN_SHUTDOWN_PROCESSES_CONFIRMED_EXITED",
                        "scope": "Diagnostic profiling may proceed; shutdown reliability is not qualified",
                        "services": rows,
                    },
                )
                print(
                    "Shutdown warning: all managed processes exited, but a service "
                    "reported failure; see shutdown-warning.json. Continuing diagnostic profiling.",
                    flush=True,
                )
            cache = runtime.paths(cfg)["stack"] / "cache"
            cache.mkdir(parents=True, exist_ok=True)
            lease = open(cache / "ort-build.owner.lock", "a+b")
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            if lease is not None:
                lease.close()
            restored = (
                services.start(active, cfg, 240)
                if active
                else services.state(list(services.ORDER), cfg, False)
            )
            runtime.write_json(output / "services-restored.json", restored)


def backend_identity(name, cfg):
    row = services.properties(name)
    port = runtime.ports(cfg)[name]
    listeners = {
        c.pid
        for c in psutil.net_connections(kind="tcp")
        if c.status == psutil.CONN_LISTEN
        and c.laddr.ip == "127.0.0.1"
        and c.laddr.port == port
    }
    if len(listeners) != 1 or not listeners <= set(row["cgroup_pids"]):
        raise RuntimeError("Native listener has no unambiguous owned process")
    stable = sorted({row["main_pid"], *listeners})
    return {
        "identity": {
            "main_pid": row["main_pid"],
            "restarts": row["NRestarts"],
            "processes": {
                str(pid): psutil.Process(pid).create_time() for pid in stable
            },
        },
        "stable_pids": stable,
        "observed_cgroup_pids": row["cgroup_pids"],
    }
