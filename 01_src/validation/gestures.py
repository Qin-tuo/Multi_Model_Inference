#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record live gesture observations; optional explicit palm-triggered stop test.

Start the normal camera application first. Camera images are saved only with
--save-preview. JPEGs are visual evidence, not the exact RGB/depth model inputs.
Use an ignored local validation directory; publish only reviewed aggregates.
"""

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import httpx
from websockets.asyncio.client import connect


class PreviewCapture:
    """Keep source/frame metadata with bounded JPEG evidence from the live app."""

    def __init__(self, output, max_bytes=64 * 1024 * 1024):
        self.directory = output.with_suffix(".frames")
        self.directory.mkdir(exist_ok=False)
        self.seen = set()
        self.report = {
            "enabled": True,
            "scope": "Web preview JPEG; lossy, no raw RGB or depth pixels. "
            "Pair only by equal source_epoch and frame_id, never by list position.",
            "byte_limit": max_bytes,
            "bytes": 0,
            "limit_rejections": 0,
            "invalid_frames": 0,
            "frames": [],
        }

    def write(self, event):
        frame = event.get("frame")
        if not isinstance(frame, dict):
            self.report["invalid_frames"] += 1
            return
        epoch, index = frame.get("source_epoch"), frame.get("frame_id")
        if not isinstance(epoch, str) or not isinstance(index, int) or index < 0:
            self.report["invalid_frames"] += 1
            return
        key = (epoch, index)
        if key in self.seen:
            return
        encoded = event.get("data")
        # Bound decoding as well as total stored bytes; never truncate a JPEG.
        if not isinstance(encoded, str) or len(encoded) > 2 * 1024 * 1024:
            self.report["invalid_frames"] += 1
            return
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError:
            self.report["invalid_frames"] += 1
            return
        if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
            self.report["invalid_frames"] += 1
            return
        if (
            len(self.seen) >= 3000
            or self.report["bytes"] + len(data) > self.report["byte_limit"]
        ):
            self.report["limit_rejections"] += 1
            return
        path = self.directory / f"{len(self.seen):06d}.jpg"
        with path.open("xb") as stream:
            stream.write(data)
        self.seen.add(key)
        self.report["bytes"] += len(data)
        self.report["frames"].append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "frame": frame,
                "received_monotonic": time.monotonic(),
            }
        )


async def probe(args):
    if not 1 <= args.seconds <= 300:
        raise ValueError("Duration must be 1–300 seconds")
    if args.output.exists():
        raise FileExistsError("Keep previous observation evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview = (
        PreviewCapture(args.output) if getattr(args, "save_preview", False) else None
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Live perception/event observation; not annotated gesture accuracy or physical audio",
        "observer_clock": "Local probe monotonic clock; compare with board timestamps only when run on the board.",
        "samples": [],
        "actions": [],
        "results": {},
        "prime_stop_test_requested": args.prime_stop_test,
    }
    report["preview"] = preview.report if preview else {"enabled": False}
    seen = set()
    known_turns = set()
    primed = None
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=5, trust_env=False
    ) as client:
        async with connect(
            args.base_url.replace("http://", "ws://", 1) + "/ws", max_size=2**21
        ) as ws:

            async def receive():
                async for raw in ws:
                    event = json.loads(raw)
                    if event.get("type") == "overflow":
                        report["transport_overflow"] = event
                        raise RuntimeError("Live event stream overflowed; evidence is incomplete")
                    if preview and event.get("type") == "frame":
                        preview.write(event)
                    if event.get("type") == "gesture":
                        report["actions"].append(event)
                        for key in ("turn_id", "target_turn_id"):
                            if event.get(key):
                                known_turns.add(event[key])
                        print(
                            json.dumps(
                                {
                                    "gesture": event["event"]["kind"],
                                    "accepted": event["accepted"],
                                    "reason": event["reason"],
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )

            receiver = asyncio.create_task(receive())
            try:
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline:
                    if receiver.done():
                        receiver.result()
                        raise RuntimeError("WebSocket ended during observation")
                    state = (await client.get("/api/state")).raise_for_status().json()
                    scene = state["perception"]
                    gestures = state["gestures"]
                    frame = scene.get("frame")
                    if frame:
                        key = (frame["source_epoch"], frame["frame_id"])
                        if key not in seen:
                            seen.add(key)
                            report["samples"].append(
                                {
                                    "perception": scene,
                                    "gestures": gestures,
                                    "observed_monotonic": time.monotonic(),
                                    "active_turn": state["active_turn"],
                                    "phase": state["phase"],
                                }
                            )
                    if (
                        args.prime_stop_test
                        and primed is None
                        and state["active_turn"] is None
                    ):
                        if any(
                            t.get("open_palm")
                            and not t.get("latched")
                            and t.get("stationary_seconds", 0) < 0.5
                            for t in gestures.get("tracks", [])
                        ):
                            response = (
                                (
                                    await client.post(
                                        "/api/chat",
                                        json={
                                            "text": "请从一数到一百，每个数字独立写一行，不要省略。"
                                        },
                                    )
                                )
                                .raise_for_status()
                                .json()
                            )
                            primed = response["turn_id"]
                            known_turns.add(primed)
                            report["primed_turn"] = primed
                            print(
                                "Stop-test request started on newly raised palm",
                                flush=True,
                            )
                    result = state.get("last_result")
                    if (
                        result
                        and result["turn_id"] in known_turns
                        and result.get("resources_released")
                    ):
                        report["results"][result["turn_id"]] = result
                    await asyncio.sleep(0.1)
                report["status"] = "OBSERVED"
            except Exception as exc:
                report.update(status="FAILED", error=str(exc))
                raise
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
                with args.output.open("x") as stream:
                    stream.write(
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "distinct_frames": len(seen),
                "events": len(report["actions"]),
                "hand_positive_frames": sum(
                    bool(s["perception"]["hands"]) for s in report["samples"]
                ),
                "completed_results": len(report["results"]),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save up to 64 MiB of live JPEG previews locally with source/frame metadata; not raw model inputs",
    )
    parser.add_argument(
        "--prime-stop-test",
        action="store_true",
        help="Submit a counting request when a fresh open palm first appears, to test cancellation while it remains held",
    )
    asyncio.run(probe(parser.parse_args()))


if __name__ == "__main__":
    main()
