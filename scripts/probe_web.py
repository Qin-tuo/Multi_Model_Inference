#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Functional probe against an already running headless web application."""
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse
import httpx
from websockets.asyncio.client import connect


async def probe(args):
    endpoint = urlparse(args.base_url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Probe requires a local HTTP endpoint")
    async with httpx.AsyncClient(base_url=args.base_url, timeout=10, trust_env=False) as client:
        state = (await client.get("/api/state")).raise_for_status().json()
        if not state["ready"]: raise RuntimeError("Application worker not ready")
        frames = []; completion = None
        async with connect(args.base_url.replace("http://", "ws://", 1) + "/ws", max_size=2**21) as ws:
            accepted = (await client.post("/api/chat", json={"text": args.text, "use_image": args.camera})).raise_for_status().json()
            async with asyncio.timeout(150):
                while True:
                    event = json.loads(await ws.recv())
                    if event["type"] == "frame": frames.append(event["frame"])
                    if event.get("turn_id") == accepted["turn_id"] and event["type"] == "done":
                        completion = event; break
        final = (await client.get("/api/state")).raise_for_status().json()
        result = {"scope": "functional smoke, not a performance baseline", "completion": completion,
                  "observed_frame_count": len(frames), "last_result": final["last_result"],
                  "audio_enabled": final["audio_enabled"], "face": final["face"]}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if completion["status"] != "complete": raise RuntimeError("Request did not complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--text", default="只回答：连接成功")
    parser.add_argument("--output", type=Path)
    asyncio.run(probe(parser.parse_args()))


if __name__ == "__main__": main()
