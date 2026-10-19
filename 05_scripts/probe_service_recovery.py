#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inject owned systemd backend crashes during real model work and verify recovery.

Requires an isolated running no-camera/file-audio application with its own DB.
Only the three named mmassistant backend units may be interrupted. Explicit CLI
opt-in is required because this affects their shared local model endpoints.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid

import httpx
import psutil
from websockets.asyncio.client import connect


UNITS = {
    "llm": "mmassistant-llama.service",
    "asr": "mmassistant-asr.service",
    "tts": "mmassistant-speech.service",
}


async def systemctl(*args):
    child = await asyncio.create_subprocess_exec(
        "systemctl",
        "--user",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(child.communicate(), 30)
    if child.returncode:
        raise RuntimeError(
            err.decode().strip() or f"systemctl exited {child.returncode}"
        )
    return out.decode().strip()


async def unit_state(unit):
    text = await systemctl(
        "show",
        unit,
        "-p",
        "ActiveState",
        "-p",
        "SubState",
        "-p",
        "MainPID",
        "-p",
        "NRestarts",
        "-p",
        "Result",
    )
    return dict(line.split("=", 1) for line in text.splitlines())


async def kill_backend(kind, unit):
    """Kill only the verified native listener, with a PID-reuse-safe handle."""
    main = int((await unit_state(unit))["MainPID"])
    owned = {p.pid for p in psutil.Process(main).children(recursive=True)}
    group = await systemctl("show", unit, "-p", "ControlGroup", "--value")
    port = {"llm": 8088, "asr": 8089, "tts": 8621}[kind]
    listeners = {
        c.pid
        for c in psutil.net_connections(kind="tcp")
        if c.status == psutil.CONN_LISTEN and c.laddr.port == port
    }
    if len(listeners) != 1 or not listeners <= owned:
        raise RuntimeError("Native listener ownership is ambiguous")
    pid = listeners.pop()
    handle = os.pidfd_open(pid)
    try:
        groups = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        if not any(line.endswith(":" + group) for line in groups):
            raise RuntimeError("Native listener moved outside owned unit")
        if int((await unit_state(unit))["MainPID"]) != main:
            raise RuntimeError("Unit restarted before injection")
        signal.pidfd_send_signal(handle, signal.SIGKILL)
    finally:
        os.close(handle)
    return {
        "pid": pid,
        "unit_main_pid": main,
        "cgroup": group,
        "port": port,
        "signal": "SIGKILL",
        "target": "verified native listener via pidfd",
    }


async def probe(args):
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": __doc__,
        "fault": "SIGKILL of verified native listener via pidfd; systemd Restart=on-failure",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "status": "RUNNING",
        "cases": [],
        "cleanup": {},
    }
    touched = set()

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    async with httpx.AsyncClient(timeout=3, trust_env=False) as client:

        async def state():
            return (
                (await client.get(args.base_url + "/api/state"))
                .raise_for_status()
                .json()
            )

        async def wait(predicate, seconds=90):
            async with asyncio.timeout(seconds):
                while True:
                    value = await predicate()
                    if value:
                        return value
                    await asyncio.sleep(0.02)

        async def idle():
            return (await state())["active_turn"] is None

        async def ready_state():
            try:
                value = await state()
                return value if value.get("ready") else None
            except httpx.HTTPError:
                return None

        try:
            initial = await wait(ready_state, 30)
        except BaseException as exc:
            report.update(status="PREFLIGHT_FAILED", error=repr(exc))
            save()
            raise
        cfg = initial["effective_config"]
        assert cfg["llm"]["base_url"] == "http://127.0.0.1:8088"
        assert cfg["stt"]["native_base_url"] == "http://127.0.0.1:8089"
        assert (
            cfg["stt"]["base_url"]
            == cfg["tts"]["base_url"]
            == "http://127.0.0.1:8621/v1"
        )
        assert (
            not initial["camera_enabled"]
            and cfg["audio"]["input_mode"] == cfg["audio"]["output_mode"] == "file"
        )
        assert (
            initial["actions"]["enabled"]
            and initial["actions"]["notes"] == 0
            and not initial["actions"]["reminders"]
        )
        assert (
            "reachy-r3-recovery-" in cfg["actions"]["database"]
            and not initial["active_turn"]
        )
        for unit in UNITS.values():
            value = await unit_state(unit)
            assert value["ActiveState"] == "active", value
        report["effective_config"] = cfg
        events = []

        async def busy(kind):
            if kind == "tts":
                value = (
                    (await client.get(cfg["tts"]["base_url"] + "/capabilities"))
                    .raise_for_status()
                    .json()
                )
                return value["tts"]["concurrency"]["active"] > 0
            url = (
                cfg["stt"]["native_base_url"]
                if kind == "asr"
                else cfg["llm"]["base_url"]
            )
            return any(
                s["is_processing"]
                for s in (await client.get(url + "/slots")).raise_for_status().json()
            )

        async def recovered(kind):
            try:
                if await busy(kind):
                    return False
                if kind == "llm":
                    values = (
                        (await client.get(cfg["llm"]["base_url"] + "/v1/models"))
                        .raise_for_status()
                        .json()
                    )
                    return cfg["llm"]["model"] in {m["id"] for m in values["data"]}
                caps = (
                    (await client.get(cfg["tts"]["base_url"] + "/capabilities"))
                    .raise_for_status()
                    .json()
                )
                value = caps[kind]
                return (
                    value["ready"]
                    and value["model_id"]
                    == cfg["stt" if kind == "asr" else "tts"]["model"]
                    and value["concurrency"]["active"] == 0
                )
            except (httpx.HTTPError, KeyError, ValueError):
                return False

        async def ask(text, request_id=None):
            body = {"text": text}
            if request_id:
                body["request_id"] = request_id
            return (
                (await client.post(args.base_url + "/api/chat", json=body))
                .raise_for_status()
                .json()["turn_id"]
            )

        async def complete(key):
            await wait(idle)
            value = (await state())["last_result"]
            assert (
                value["turn_id"] == key
                and value["status"] == "complete"
                and value["resources_released"]
            ), value
            assert (
                await client.get(args.base_url + value["audio_url"])
            ).status_code == 200
            return value

        async with connect(
            args.base_url.replace("http://", "ws://", 1) + "/ws", max_size=2**21
        ) as ws:

            async def receive():
                async for raw in ws:
                    event = json.loads(raw)
                    if event.get("turn_id"):
                        events.append(event)
                        if len(events) > 2000:
                            raise RuntimeError(
                                "Unexpected unbounded diagnostic event volume"
                            )

            receiver = asyncio.create_task(receive())
            try:
                for scenario, kind in [
                    ("llm_inference", "llm"),
                    ("asr_inference", "asr"),
                    ("tts_synthesis", "tts"),
                    ("llm_after_note_commit", "llm"),
                ]:
                    await wait(idle)
                    assert await recovered(kind)
                    case = {
                        "scenario": scenario,
                        "unit": UNITS[kind],
                        "before": await unit_state(UNITS[kind]),
                        "status": "RUNNING",
                    }
                    report["cases"].append(case)
                    save()
                    before_session = (await state())["session"]["retained_turns"]
                    if kind == "asr":
                        response = await client.post(
                            args.base_url + "/api/audio",
                            content=args.input.read_bytes(),
                            headers={"Content-Type": "audio/wav"},
                        )
                        key = response.raise_for_status().json()["turn_id"]
                    elif scenario == "llm_after_note_commit":
                        request = uuid.uuid4().hex
                        prompt = "请创建笔记，标题为故障恢复验证，内容为已提交的笔记在模型服务退出后仍需保留。保存完成后详细解释这个原则。"
                        key = await ask(prompt, request)

                        async def receipt():
                            return next(
                                (
                                    e["receipt"]
                                    for e in events
                                    if e["turn_id"] == key
                                    and e["type"] == "action_receipt"
                                    and e["receipt"].get("committed")
                                ),
                                None,
                            )

                        case["committed_before_fault"] = await wait(receipt, 30)
                    else:
                        key = await ask(
                            "请详细解释十种常见的数据结构，每项说明用途和优缺点。"
                        )
                    case["turn_id"] = key

                    async def observed_busy():
                        return await busy(kind)

                    await wait(observed_busy, 30)
                    case["busy_observed"] = True
                    touched.add(kind)
                    case["fault_monotonic"] = time.monotonic()
                    case["injection"] = await kill_backend(kind, UNITS[kind])
                    case["after_kill"] = await unit_state(UNITS[kind])
                    save()

                    async def finalized():
                        current = await state()
                        result = current.get("last_result")
                        return current if result and result["turn_id"] == key else None

                    failed = await wait(finalized, 30)
                    case["during_recovery"] = failed
                    assert failed["last_result"]["status"] == "failed", failed[
                        "last_result"
                    ]
                    if failed["active_turn"]:
                        blocked = await client.post(
                            args.base_url + "/api/chat",
                            json={"text": "资源未释放时不应接受"},
                        )
                        assert blocked.status_code == 409
                        case["admission_while_unreleased"] = 409
                    await wait(lambda: recovered(kind), 120)
                    await wait(idle, 30)
                    final = (await state())["last_result"]
                    case["failed_result"] = final
                    assert (
                        final["turn_id"] == key
                        and final["status"] == "failed"
                        and final["resources_released"]
                    )
                    assert (
                        "audio_url" not in final
                        and (
                            await client.get(args.base_url + "/api/audio/" + key)
                        ).status_code
                        == 404
                    )
                    assert not list(
                        Path(cfg["audio"]["output_directory"]).glob("*.partial")
                    )
                    assert (await state())["session"][
                        "retained_turns"
                    ] == before_session
                    case["after"] = await unit_state(UNITS[kind])
                    assert case["after"]["MainPID"] != case["before"]["MainPID"]
                    assert int(case["after"]["NRestarts"]) > int(
                        case["before"]["NRestarts"]
                    )
                    case["fault_to_released_seconds"] = (
                        time.monotonic() - case["fault_monotonic"]
                    )
                    if scenario == "llm_after_note_commit":
                        assert (await state())["actions"]["notes"] == 1
                        replay = await complete(await ask(prompt, request))
                        assert replay["replayed"] and "model_rounds" not in replay
                        assert (
                            replay["actions"][0]["result"]
                            == case["committed_before_fault"]["result"]
                        )
                        assert (await state())["actions"]["notes"] == 1
                        case["same_request_replay"] = replay
                    case["next_turn"] = await complete(
                        await ask("请只回复：故障恢复成功。")
                    )
                    case["events"] = [e for e in events if e["turn_id"] == key]
                    case["status"] = "PASSED"
                    save()
                    print(
                        json.dumps(
                            {
                                "scenario": scenario,
                                "status": "PASSED",
                                "recovery_seconds": case["fault_to_released_seconds"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                report["status"] = "PASSED_WITH_DECLARED_SCOPE"
            except BaseException as exc:
                if report["cases"] and report["cases"][-1]["status"] == "RUNNING":
                    report["cases"][-1]["status"] = "FAILED"
                report.update(status="FAILED", error=repr(exc))
                raise
            finally:
                # Ensure owned backends are restored even if a probe assertion fails.
                for kind in touched:
                    try:
                        await systemctl("start", UNITS[kind])
                        await wait(lambda: recovered(kind), 120)
                        report["cleanup"][kind] = {
                            "restored": True,
                            "unit": await unit_state(UNITS[kind]),
                        }
                    except Exception as exc:
                        report["cleanup"][kind] = {
                            "restored": False,
                            "error": repr(exc),
                        }
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
                save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inject-managed-service-crashes", action="store_true", required=True
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8093")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(probe(parser.parse_args()))
