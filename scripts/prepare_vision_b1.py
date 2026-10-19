#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch pinned B1 candidates without installing or switching any runtime model."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "03_models/candidates/vision-b1.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(role, entry, check_only=False):
    path = ROOT / entry["path"]
    if not path.resolve().is_relative_to(ROOT / "05_runtime/models/b1"):
        raise ValueError("Candidate path escapes the B1 model directory")
    if path.exists():
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Existing candidate differs from its lock: " + role)
        return {"role": role, "verified": True, "downloaded": False}
    if check_only:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        entry["repository"] + "/resolve/" + entry["revision"] + "/" + entry["filename"]
    )
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(
            "Inspect the interrupted download before retrying: " + str(partial)
        )
    try:
        received, report_at = 0, time.monotonic()
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            partial.open("xb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                received += len(chunk)
                if received > entry["bytes"]:
                    raise ValueError("Downloaded candidate exceeds its recorded size")
                if time.monotonic() - report_at >= 10:
                    print(f"{role}: {received / 2**20:.1f} MiB downloaded", flush=True)
                    report_at = time.monotonic()
        if received != entry["bytes"] or digest(partial) != entry["sha256"]:
            raise ValueError("Downloaded candidate checksum/size mismatch: " + role)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    # Keep the exact model card beside the candidate; do not execute upstream code.
    card_url = entry["repository"] + "/raw/" + entry["revision"] + "/README.md"
    with urllib.request.urlopen(card_url, timeout=30) as response:
        card = response.read()
    (path.parent / "MODEL_CARD.md").write_bytes(card)
    return {
        "role": role,
        "verified": True,
        "downloaded": True,
        "model_card_sha256": hashlib.sha256(card).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("depth", "body_hand_pose"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    entries = json.loads(MANIFEST.read_text())["models"]
    for role in [args.role] if args.role else entries:
        print(json.dumps(prepare(role, entries[role], args.check_only)), flush=True)


if __name__ == "__main__":
    main()
