#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare the locked YuNet artifact outside the repository; no runtime download."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_src"))
from project_paths import load_artifact_manifest
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entry = load_artifact_manifest(root / "03_models/perception.json")["face"]
    folder = Path(entry["directory"]).expanduser()
    for name, identity in entry["files"].items():
        path = folder / name
        if (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
        ):
            continue
        if args.check_only:
            raise RuntimeError(f"Missing or mismatched face artifact: {name}")
        folder.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(identity["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != identity["sha256"]:
            raise ValueError(f"Downloaded artifact hash mismatch: {name}")
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(data)
        temporary.replace(path)
    print(
        json.dumps(
            {"verified": True, "directory": str(folder), "revision": entry["revision"]}
        )
    )


if __name__ == "__main__":
    main()
