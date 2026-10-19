#!/usr/bin/env python3
"""E1: real TCP app requests, fixed camera source, native models and local tools.

The stateful workload has one explicit warmup sequence, not repeated hidden
mutating commands. Arrival order, history reset boundaries and all rejections
are recorded. Physical capture/playback and browser paint are separate scopes.
"""

import argparse
import asyncio
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
import wave

import httpx
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_src"))
# Shared native-process identity checks live with the functional probes.
sys.path.insert(0, str(ROOT / "05_scripts"))
from benchmarks.baseline_support import Events, Resources, environment, validate  # noqa: E402
from deploy import runtime, services  # noqa: E402
from probe_coexistence import backend_identity  # noqa: E402


class Server:
    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, out
        self.child = self.log = None
        self.serial = 0
        self.runs = []

    def start(self):
        self.serial += 1
        folder = self.out / f"server-{self.serial}"
        folder.mkdir()
        self.log = (folder / "process.log").open("w")
        self.child = subprocess.Popen(
            [
                sys.executable,
                ROOT / "01_src/benchmarks/application_server.py",
                "--config",
                self.out / "settings.json",
                "--output",
                folder,
            ],
            cwd=ROOT,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        row = {
            "pid": self.child.pid,
            "directory": str(folder),
            "started_at": runtime.stamp(),
        }
        self.runs.append(row)
        deadline = time.monotonic() + 120
        while True:
            if self.child.poll() is not None:
                raise RuntimeError("Benchmark application exited at startup")
            try:
                health = services.health("app", self.cfg)
                if health["ready"] and services.owns_port(
                    self.cfg.web.port, [self.child.pid]
                ):
                    row["ready"] = health
                    return
            except (httpx.HTTPError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Benchmark application did not become ready")
            time.sleep(0.25)

    def stop(self):
        if self.child is None:
            return
        if self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGINT)
            try:
                self.child.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait(10)
                self.runs[-1]["forced_stop"] = True
        self.runs[-1]["exit_code"] = self.child.returncode
        self.log.close()
        if self.child.returncode != 0:
            raise RuntimeError("Benchmark app shutdown was not clean")


def notes(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        return {
            row[0]: {"title": row[1], "body": row[2]}
            for row in connection.execute("SELECT id,title,body FROM reachy_notes")
        }


async def exercise(cfg, server, out, report):
    base = f"http://127.0.0.1:{cfg.web.port}"
    inputs = Path.home() / "AI_Stack/datasets/reachy-b0-sw-v1"
    text_cases = [
        c
        for c in json.loads(
            (ROOT / "06_benchmarks/corpora/reachy-b0-llm-v1.json").read_text()
        )["cases"]
        if c["kind"] == "text"
    ]
    speech_cases = json.loads(
        (ROOT / "06_benchmarks/corpora/reachy-b0-speech-v1.json").read_text()
    )["cases"]
    identities = {
        name: backend_identity(name, cfg) for name in ("llama", "asr", "speech")
    }
    groups = {name: row["stable_pids"] for name, row in identities.items()}
    groups["app"] = [server.child.pid]
    resources = Resources(out, groups).start()
    report["backend_before"] = identities
    events = Events(out / "client-events.jsonl")
    states = Events(out / "state-samples.jsonl")
    records = defaultdict(list)
    changed = asyncio.Condition()
    connected = asyncio.Event()
    stop = asyncio.Event()
    phase = "warmup"
    frames = 0
    sampler_errors = []

    def save():
        runtime.write_json(out / "report.json", report)

    try:
        async with httpx.AsyncClient(
            base_url=base,
            timeout=5,
            trust_env=False,
            limits=httpx.Limits(max_connections=24, max_keepalive_connections=6),
        ) as client:

            async def state():
                return (await client.get("/api/state")).raise_for_status().json()

            async def idle(seconds=180):
                deadline = time.monotonic() + seconds
                while True:
                    value = await state()
                    if value["active_turn"] is None and not value["voice"]["pending"]:
                        return value
                    if time.monotonic() > deadline:
                        raise RuntimeError("App did not return to idle")
                    await asyncio.sleep(0.1)

            async def has_event(key, predicate, seconds=180):
                async with asyncio.timeout(seconds):
                    async with changed:
                        await changed.wait_for(
                            lambda: any(predicate(e) for e in records[key])
                        )
                return next(e for e in records[key] if predicate(e))

            async def sampler():
                origin = time.monotonic()
                index = 0
                while not stop.is_set():
                    due = origin + index / 10
                    await asyncio.sleep(max(0, due - time.monotonic()))
                    tick = time.monotonic()
                    row = {
                        "sample": index,
                        "phase": phase,
                        "scheduled": due,
                        "sampled": tick,
                    }
                    index += 1
                    try:
                        if tick - due >= 0.1:
                            row.update(
                                missing="observer_missed_deadline", latest_age_ms=None
                            )
                        else:
                            value = await state()
                            scene = value["perception"]
                            frame = scene.get("frame")
                            row.update(
                                camera=value["camera"],
                                web=value["web"],
                                phase_state=value["phase"],
                                session=value["session"],
                                pending=value["pending"],
                                actions=value["actions"],
                                perception={
                                    k: scene.get(k)
                                    for k in (
                                        "ready",
                                        "error",
                                        "processed",
                                        "failures",
                                        "skipped_capture_frames",
                                        "elapsed_ms",
                                        "stages",
                                    )
                                },
                                latest_age_ms=(
                                    time.monotonic() - frame["received_monotonic"]
                                )
                                * 1000
                                if scene["ready"] and frame
                                else None,
                            )
                    except Exception as exc:
                        row.update(missing=repr(exc), latest_age_ms=None)
                        sampler_errors.append({"sample": index, "error": repr(exc)})
                    states.write(**row)

            async with connect(
                base.replace("http://", "ws://") + "/ws", max_size=2**21
            ) as ws:

                async def receive():
                    nonlocal frames
                    async for raw in ws:
                        value = json.loads(raw)
                        tick = time.monotonic()
                        if value["type"] == "capabilities":
                            connected.set()
                        if value["type"] == "frame":
                            frames += 1
                            events.write(
                                kind="frame",
                                received=tick,
                                phase=phase,
                                frame=value.get("frame"),
                            )
                            continue
                        events.write(
                            kind="websocket", received=tick, phase=phase, message=value
                        )
                        if key := value.get("turn_id"):
                            async with changed:
                                records[key].append({"received": tick, **value})
                                changed.notify_all()

                reader = asyncio.create_task(receive())
                observer = asyncio.create_task(sampler())

                async def submit(kind, name, text=None, audio=None, image=False):
                    await idle()
                    request_id = uuid.uuid5(
                        uuid.NAMESPACE_URL, str(out) + "/" + name
                    ).hex
                    start = time.monotonic()
                    if audio:
                        response = await client.post(
                            "/api/audio",
                            params={"use_image": image, "request_id": request_id},
                            content=audio.read_bytes(),
                            headers={"Content-Type": "audio/wav"},
                        )
                    else:
                        response = await client.post(
                            "/api/chat",
                            json={
                                "text": text,
                                "use_image": image,
                                "request_id": request_id,
                            },
                        )
                    row = {
                        "class": kind,
                        "case": name,
                        "phase": phase,
                        "request_id": request_id,
                        "text": text,
                        "audio": str(audio) if audio else None,
                        "use_image": image,
                        "submitted": start,
                        "response_received": time.monotonic(),
                        "http_status": response.status_code,
                    }
                    if response.status_code == 202:
                        row.update(response.json())
                    else:
                        row["response"] = response.text
                    report["requests"].append(row)
                    save()
                    return row

                async def finish(row, cancelled=False):
                    if row["http_status"] != 202:
                        row["outcome"] = "rejected"
                        report["checks_failed"].append(
                            {
                                "case": row["case"],
                                "unexpected_admission_rejection": row["http_status"],
                            }
                        )
                        return row
                    key = row["turn_id"]
                    done = await has_event(key, lambda e: e["type"] == "done")
                    free = await has_event(
                        key,
                        lambda e: e["type"] == "status" and e.get("stage") == "idle",
                    )
                    row.update(
                        outcome=done["status"],
                        done=done,
                        idle_received=free["received"],
                        submit_to_done_seconds=done["received"] - row["submitted"],
                        submit_to_idle_seconds=free["received"] - row["submitted"],
                    )
                    token = next(
                        (
                            e
                            for e in records[key]
                            if e["type"] == "token" and e.get("text")
                        ),
                        None,
                    )
                    row["first_client_content_seconds"] = (
                        token["received"] - row["submitted"] if token else None
                    )
                    snapshot = await state()
                    row["result"] = (
                        snapshot["last_result"]
                        if (snapshot.get("last_result") or {}).get("turn_id") == key
                        else None
                    )
                    if done.get("audio_url"):
                        downloaded = time.monotonic()
                        reply = (
                            (await client.get(done["audio_url"]))
                            .raise_for_status()
                            .content
                        )
                        with wave.open(io.BytesIO(reply), "rb") as wav:
                            shape = {
                                "rate": wav.getframerate(),
                                "channels": wav.getnchannels(),
                                "width": wav.getsampwidth(),
                                "frames": wav.getnframes(),
                            }
                        if (shape["rate"], shape["channels"], shape["width"]) != (
                            16000,
                            1,
                            2,
                        ):
                            raise RuntimeError("Unexpected complete-app WAV format")
                        destination = out / "downloads" / f"{key}.wav"
                        destination.parent.mkdir(exist_ok=True)
                        destination.write_bytes(reply)
                        row["download"] = {
                            "seconds": time.monotonic() - downloaded,
                            "sha256": hashlib.sha256(reply).hexdigest(),
                            "format": shape,
                        }
                    if cancelled:
                        row["cancelled_audio_status"] = (
                            await client.get("/api/audio/" + key)
                        ).status_code
                        index = next(
                            i
                            for i, e in enumerate(records[key])
                            if e["type"] == "cancelled"
                        )
                        row["late_output"] = any(
                            e["type"] in {"token", "transcript"}
                            for e in records[key][index + 1 :]
                        )
                    expected = "cancelled" if cancelled else "complete"
                    if row["outcome"] != expected:
                        report["checks_failed"].append(
                            {
                                "case": row["case"],
                                "expected": expected,
                                "observed": row["outcome"],
                            }
                        )
                    save()
                    return row

                async def clear():
                    value = await idle()
                    (
                        await client.post(
                            "/api/session/clear",
                            json={"session_id": value["session"]["id"]},
                        )
                    ).raise_for_status()

                async def busy(role):
                    if role == "tts":
                        value = (
                            (await client.get(cfg.tts.base_url + "/capabilities"))
                            .raise_for_status()
                            .json()
                        )
                        return value["tts"]["concurrency"]["active"] > 0
                    base_url = (
                        cfg.stt.native_base_url if role == "asr" else cfg.llm.base_url
                    )
                    return any(
                        s["is_processing"]
                        for s in (await client.get(base_url + "/slots"))
                        .raise_for_status()
                        .json()
                    )

                try:
                    await asyncio.wait_for(connected.wait(), 10)
                    # Let static-image gesture tracks settle while no request is active.
                    await asyncio.sleep(20)
                    report["settled_state"] = await state()
                    for index in range(5):
                        await finish(
                            await submit(
                                "warmup",
                                f"warm-text-{index}",
                                text="只回答：预热完成。",
                            )
                        )
                        await finish(
                            await submit(
                                "warmup",
                                f"warm-audio-{index}",
                                audio=inputs / "zh-20.wav",
                            )
                        )
                        await finish(
                            await submit(
                                "warmup",
                                f"warm-image-{index}",
                                audio=inputs / "zh-03.wav",
                                image=True,
                            )
                        )
                    if notes(cfg.actions.database):
                        raise RuntimeError(
                            "Non-mutating warmup unexpectedly created a note"
                        )
                    for kind in ("text", "audio", "audio_image"):
                        await clear()
                        phase = resources.phase = kind
                        for index in range(20):
                            print(f"E1 {kind}: {index + 1}/20", flush=True)
                            row = await submit(
                                kind,
                                f"{kind}-{index + 1:02d}",
                                text=text_cases[index]["prompt"]
                                if kind == "text"
                                else None,
                                audio=inputs / (speech_cases[index]["id"] + ".wav")
                                if kind != "text"
                                else None,
                                image=kind == "audio_image",
                            )
                            await finish(row)
                    await clear()
                    phase = resources.phase = "actions"
                    for index in range(1, 21):
                        print(f"E1 actions: {index}/20", flush=True)
                        title = f"软件基线第{index}条"
                        body = f"第{index}条记录用于验证本地笔记保存。"
                        row = await submit(
                            "actions",
                            f"action-{index:02d}",
                            text=f"创建一条笔记，标题原样使用“{title}”，正文原样使用“{body}”。请调用工具保存后简短回复。",
                        )
                        await finish(row)
                        if row["http_status"] != 202:
                            continue
                        receipts = [
                            e["receipt"]
                            for e in records[row["turn_id"]]
                            if e["type"] == "action_receipt"
                            and e["receipt"]["name"] == "create_note"
                            and e["receipt"]["committed"]
                        ]
                        persisted = notes(cfg.actions.database)
                        row["verified_notes"] = []
                        for receipt in receipts:
                            note = receipt["result"]
                            actual = persisted.get(note["note_id"])
                            row["verified_notes"].append(
                                {
                                    "id": note["note_id"],
                                    "persisted_matches_receipt": actual
                                    == {"title": note["title"], "body": note["body"]},
                                    "matches_requested_literal": actual
                                    == {"title": title, "body": body},
                                }
                            )
                        if len(row["verified_notes"]) != 1 or not all(
                            n["persisted_matches_receipt"]
                            and n["matches_requested_literal"]
                            for n in row["verified_notes"]
                        ):
                            report["checks_failed"].append(
                                {
                                    "case": row["case"],
                                    "note_receipt_or_literal_mismatch": True,
                                }
                            )
                        save()
                    await clear()
                    phase = resources.phase = "cancellation"
                    for index in range(20):
                        target = ("asr", "llm", "tts")[index % 3]
                        print(
                            f"E1 cancellation: {index + 1}/20 during {target}",
                            flush=True,
                        )
                        row = await submit(
                            "cancellation",
                            f"cancel-{index:02d}",
                            audio=inputs / "zh-19.wav" if target == "asr" else None,
                            text=None
                            if target == "asr"
                            else "请详细介绍十种学习编程的方法，每条解释具体步骤，不执行任何工具。",
                        )
                        row["target_stage"] = target
                        if row["http_status"] != 202:
                            await finish(row)
                            continue
                        deadline = time.monotonic() + 45
                        while not await busy(target):
                            if time.monotonic() > deadline or any(
                                e["type"] == "done" for e in records[row["turn_id"]]
                            ):
                                row["target_stage_not_observed"] = True
                                report["checks_failed"].append(
                                    {
                                        "case": row["case"],
                                        "target_stage_not_observed": target,
                                    }
                                )
                                break
                            await asyncio.sleep(0.01)
                        row["cancel_sent"] = time.monotonic()
                        await ws.send(
                            json.dumps({"type": "cancel", "turn_id": row["turn_id"]})
                        )
                        await finish(
                            row,
                            cancelled=not row.get("target_stage_not_observed", False),
                        )
                        row["cancel_to_idle_seconds"] = (
                            row["idle_received"] - row["cancel_sent"]
                        )
                        if (
                            row.get("late_output")
                            or row.get("cancelled_audio_status", 404) != 404
                        ):
                            report["checks_failed"].append(
                                {"case": row["case"], "cancelled_output_leaked": True}
                            )
                    await clear()
                    phase = resources.phase = "overload"
                    incumbent = await submit(
                        "overload_incumbent",
                        "overload-incumbent",
                        text="请详细介绍十种学习编程的方法，每条解释具体步骤，不执行任何工具。",
                    )
                    busy_deadline = time.monotonic() + 45
                    while not await busy("llm"):
                        if time.monotonic() > busy_deadline or any(
                            e["type"] == "done" for e in records[incumbent["turn_id"]]
                        ):
                            raise RuntimeError("Overload incumbent never became active")
                        await asyncio.sleep(0.01)
                    origin = time.monotonic() + 0.1

                    async def arrival(index):
                        due = origin + index * 0.05
                        await asyncio.sleep(max(0, due - time.monotonic()))
                        tick = time.monotonic()
                        response = await client.post(
                            "/api/chat", json={"text": f"第{index}个并发到达测试请求"}
                        )
                        row = {
                            "index": index,
                            "scheduled": due,
                            "sent": tick,
                            "received": time.monotonic(),
                            "http_status": response.status_code,
                            "response": response.text,
                        }
                        report["overload"].append(row)

                    await asyncio.gather(*(arrival(i) for i in range(20)))
                    incumbent["cancel_sent"] = time.monotonic()
                    await ws.send(
                        json.dumps({"type": "cancel", "turn_id": incumbent["turn_id"]})
                    )
                    await finish(incumbent, cancelled=True)
                    if any(
                        r["http_status"] not in {409, 503} for r in report["overload"]
                    ):
                        report["checks_failed"].append(
                            {"unexpected_overload_admission": True}
                        )
                    phase = resources.phase = "recovery"
                    await finish(
                        await submit("recovery", "recovery", text="只回答：恢复成功。")
                    )
                    report["notes_before_restart"] = notes(cfg.actions.database)
                    report["final_state"] = await state()
                    report["frames_received"] = frames
                    report["backend_after"] = {
                        name: backend_identity(name, cfg) for name in identities
                    }
                    if any(
                        report["backend_after"][name]["identity"] != before["identity"]
                        for name, before in identities.items()
                    ):
                        report["checks_failed"].append(
                            {"native_backend_restarted": True}
                        )
                finally:
                    stop.set()
                    reader.cancel()
                    await asyncio.gather(reader, observer, return_exceptions=True)
    finally:
        resources.close()
        events.close()
        states.close()
        report["sampler_errors"] = sampler_errors
        report["monitor_errors"] = resources.errors
        save()


async def verify_restart(cfg, out, report):
    """Outside performance windows: actual API replay/query after app restart."""
    base = f"http://127.0.0.1:{cfg.web.port}"
    original = next(
        r
        for r in reversed(report["requests"])
        if r["class"] == "actions" and r.get("outcome") == "complete"
    )
    before = notes(cfg.actions.database)
    async with httpx.AsyncClient(base_url=base, timeout=5, trust_env=False) as client:
        async with connect(
            base.replace("http://", "ws://") + "/ws", max_size=2**21
        ) as ws:
            async with asyncio.timeout(15):
                while json.loads(await ws.recv())["type"] != "capabilities":
                    pass

            async def run(payload):
                response = (
                    (await client.post("/api/chat", json=payload))
                    .raise_for_status()
                    .json()
                )
                key = response["turn_id"]
                async with asyncio.timeout(180):
                    while True:
                        message = json.loads(await ws.recv())
                        if (
                            message.get("turn_id") == key
                            and message.get("type") == "status"
                            and message.get("stage") == "idle"
                        ):
                            break
                result = (
                    (await client.get("/api/state"))
                    .raise_for_status()
                    .json()["last_result"]
                )
                if (
                    result["turn_id"] != key
                    or result["status"] != "complete"
                    or not result["resources_released"]
                ):
                    raise RuntimeError("Post-restart request did not complete/release")
                wave_data = (
                    (await client.get(result["audio_url"])).raise_for_status().content
                )
                if not wave_data.startswith(b"RIFF"):
                    raise RuntimeError("Post-restart reply is not a WAV")
                return result

            replay = await run(
                {
                    "text": original["text"],
                    "request_id": original["request_id"],
                    "use_image": False,
                }
            )
            if (
                not replay.get("replayed")
                or replay.get("mode") != "receipt_replay"
                or notes(cfg.actions.database) != before
            ):
                raise RuntimeError("Idempotent replay changed persisted notes")
            query = await run(
                {"text": "请查询最近三条笔记并给出标题，不创建或修改任何笔记。"}
            )
            receipts = [
                a
                for a in query.get("actions", [])
                if a["name"] == "list_notes" and a["committed"]
            ]
            with sqlite3.connect(
                f"file:{cfg.actions.database}?mode=ro", uri=True
            ) as connection:
                expected = [
                    r[0]
                    for r in connection.execute(
                        "SELECT id FROM reachy_notes ORDER BY created_at DESC LIMIT 3"
                    )
                ]
            if (
                not receipts
                or [n["note_id"] for n in receipts[-1]["result"]["notes"]] != expected
            ):
                raise RuntimeError(
                    "Restarted model did not query the actual persisted notes"
                )
            if notes(cfg.actions.database) != before:
                raise RuntimeError("Read-only query changed notes")
    result = {
        "status": "PASS",
        "scope": "Functional verification after collection, excluded from performance statistics",
        "created_at": runtime.stamp(),
        "note_count": len(before),
        "replayed_request": original["request_id"],
        "replay": replay,
        "query": query,
        "expected_recent_note_ids": expected,
    }
    runtime.write_json(out / "restart-api-verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    _, original, _ = validate()
    before = services.state(services.ORDER, original)
    if (
        before["status"] != "READY"
        or before["services"]["app"]["health"]["phase"] != "idle"
    ):
        raise RuntimeError("Managed services must be ready and idle")
    environment(out)
    cfg = runtime.config(services.profile_path(original))
    cfg.web.port = 8093
    cfg.actions.database = str(out / "actions.sqlite")
    cfg.audio.output_directory = str(out / "audio")
    cfg.validate()
    runtime.write_json(out / "settings.json", cfg.effective())
    runtime.write_json(out / "effective-config.json", original.effective())
    fixture = ROOT / "06_benchmarks/manifests/reachy-b0-e1-input-v1.json"
    (out / "camera-input.json").write_bytes(fixture.read_bytes())
    workload = {
        "camera": "Constant frozen public street scene through production Camera, 10Hz absolute replay; no physical camera input in E1",
        "camera_fixture_manifest_sha256": runtime.sha256(fixture),
        "features": "All production modules and gestures enabled; 20 seconds of settling before requests. Ordinary-input scene has no intentional open-palm stop command. Gesture events remain recorded, not suppressed.",
        "config_changes": [
            "web.port=8093",
            "isolated actions.database",
            "isolated audio.output_directory",
        ],
        "warmup": "Five representative text, five file voice, five image+voice requests; no mutating warmup. Then clear session via public API. Stateful mutating workflows are not silently repeated as warmups.",
        "classes": {
            "text": 20,
            "audio": 20,
            "audio_image": 20,
            "actions": 20,
            "cancellation": 20,
            "overload_arrivals": 20,
        },
        "history": "Fresh app/DB, history retained within each class, cleared at class boundaries; durable state retained throughout",
        "arrival": "Serial interactive requests for normal classes; overload is twenty independently scheduled arrivals 50ms apart while a long request is active",
        "audio_input_manifest_sha256": runtime.sha256(
            ROOT / "06_benchmarks/manifests/reachy-b0-inputs-v1.json"
        ),
        "text_corpus_sha256": runtime.sha256(
            ROOT / "06_benchmarks/corpora/reachy-b0-llm-v1.json"
        ),
        "audio_corpus_sha256": runtime.sha256(
            ROOT / "06_benchmarks/corpora/reachy-b0-speech-v1.json"
        ),
        "measurement": "Client WS receipt, not browser paint; TTS first PCM estimate uses wrapper start + production first-block duration, not audible playback. Hooks/observers included in total time.",
    }
    runtime.write_json(out / "workload.json", workload)
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "before": before,
        "requests": [],
        "overload": [],
        "checks_failed": [],
        "workload_sha256": runtime.sha256(out / "workload.json"),
    }
    server = Server(cfg, out)
    error = None
    try:
        with services.operation(original):
            services.stop(list(services.ORDER), original)
            services.start(["llama", "asr", "speech"], original, 240)
        services.preflight(["app"], cfg)
        server.start()
        asyncio.run(exercise(cfg, server, out, report))
        server.stop()
        report["notes_after_process_exit"] = notes(cfg.actions.database)
        if report["notes_after_process_exit"] != report["notes_before_restart"]:
            raise RuntimeError("Persistent notes changed on app exit")
        server.start()
        report["notes_after_restart"] = notes(cfg.actions.database)
        if report["notes_after_restart"] != report["notes_before_restart"]:
            raise RuntimeError("Persistent notes changed on app restart")
        report["restart_ready"] = services.health("app", cfg)
        report["restart_api_verification"] = asyncio.run(
            verify_restart(cfg, out, report)
        )
    except BaseException as exc:
        error = exc
        report["error"] = repr(exc)
    finally:
        try:
            server.stop()
        except Exception as exc:
            report["cleanup_error"] = repr(exc)
            error = error or exc
        report["server_runs"] = server.runs
        try:
            with services.operation(original):
                report["restored"] = services.start(list(services.ORDER), original, 240)
        except Exception as exc:
            error = error or exc
            report["restore_error"] = repr(exc)
        report["status"] = "FAIL" if error else "COLLECTED_REVIEW_CHECKS"
        report["finished_at"] = runtime.stamp()
        runtime.write_json(out / "report.json", report)
    if error:
        raise error
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}))


if __name__ == "__main__":
    main()
