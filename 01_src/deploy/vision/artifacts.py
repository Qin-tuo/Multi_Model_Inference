#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check or fetch pinned current model artifacts without installing or switching any runtime model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "03_models/vision-models.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_sources(entry, check_only):
    """Restore the small pinned upstream helpers without executing their code."""
    rows = []
    for key in ("prepost_source", "postprocess_source"):
        group = entry.get(key, {})
        rows.extend(group.get("files", {}).values())
        rows.extend(group.get("license_sources", []))
    if source := entry.get("preprocessing", {}).get("source_cpp"):
        rows.append(source)
    for row in rows:
        path = ROOT / row["path"]
        if not path.resolve().is_relative_to((ROOT / "05_runtime/frameworks").resolve()):
            raise ValueError("Upstream helper path escapes the runtime framework directory")
        if path.is_file():
            if digest(path) != row["sha256"]:
                raise ValueError("Upstream helper checksum mismatch: " + str(path))
            continue
        if check_only:
            raise FileNotFoundError(path)
        with urllib.request.urlopen(row["url"], timeout=30) as response:
            data = response.read(2 * 1024**2 + 1)
        if len(data) > 2 * 1024**2 or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Downloaded upstream helper differs from its lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        owned = False
        try:
            with partial.open("xb") as stream:
                owned = True
                stream.write(data)
            os.link(partial, path)
        finally:
            if owned:
                partial.unlink(missing_ok=True)


def prepare(role, entry, check_only=False):
    path = ROOT / entry["path"]
    if not path.resolve().is_relative_to((ROOT / "05_runtime/models").resolve()):
        raise ValueError("Candidate path escapes the runtime model directory")
    if check_only and not path.is_file():
        raise FileNotFoundError(path)
    if entry.get("model_card_sha256"):
        card_path = ROOT / entry.get("model_card_path", str(path.parent / "MODEL_CARD.md"))
        if not card_path.exists():
            if check_only:
                raise FileNotFoundError(card_path)
            card_url = (
                entry.get("model_card_url")
                or entry["repository"] + "/raw/" + entry["revision"] + "/README.md"
            )
            with urllib.request.urlopen(card_url, timeout=30) as response:
                card = response.read(256 * 1024 + 1)
            if (
                len(card) > 256 * 1024
                or hashlib.sha256(card).hexdigest() != entry["model_card_sha256"]
            ):
                raise ValueError("Model card differs from its recorded source")
            path.parent.mkdir(parents=True, exist_ok=True)
            with card_path.open("xb") as out:
                out.write(card)
        if digest(card_path) != entry["model_card_sha256"]:
            raise ValueError("Existing model card differs from its lock")
    prepare_sources(entry, check_only)
    if path.exists():
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Existing candidate differs from its lock: " + role)
        return {
            "role": role,
            "verified": True,
            "downloaded": False,
            "model_card_sha256": entry.get("model_card_sha256"),
        }
    if entry.get("artifact_kind") == "local_export" or "source_checkpoint" in entry:
        raise FileNotFoundError(
            "Export the pinned native weights with scripts/build.py export-* first: "
            + str(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        entry["repository"] + "/resolve/" + entry["revision"] + "/" + entry["filename"]
    )
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(
            "Inspect the interrupted download before retrying: " + str(partial)
        )
    owns_partial = False
    try:
        received, report_at = 0, time.monotonic()
        with partial.open("xb") as out:
            owns_partial = True
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    received += len(chunk)
                    if received > entry["bytes"]:
                        raise ValueError(
                            "Downloaded candidate exceeds its recorded size"
                        )
                    if time.monotonic() - report_at >= 10:
                        print(
                            f"{role}: {received / 2**20:.1f} MiB downloaded", flush=True
                        )
                        report_at = time.monotonic()
        if received != entry["bytes"] or digest(partial) != entry["sha256"]:
            raise ValueError("Downloaded candidate checksum/size mismatch: " + role)
        # Atomic publication must not overwrite another writer's artifact.
        os.link(partial, path)
        partial.unlink()
    except BaseException:
        if owns_partial:
            partial.unlink(missing_ok=True)
        raise
    return {
        "role": role,
        "verified": True,
        "downloaded": True,
        "model_card_sha256": entry.get("model_card_sha256"),
    }


def main():
    entries = json.loads(MANIFEST.read_text())["models"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(entries))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    selected = (
        [args.role]
        if args.role
        else [
            role
            for role, item in entries.items()
            if item.get("phase") != "DEFERRED_EXPERIMENT"
        ]
    )
    for role in selected:
        print(json.dumps(prepare(role, entries[role], args.check_only)), flush=True)


if __name__ == "__main__":
    main()
