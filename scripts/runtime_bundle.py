#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inventory, verify, pack and restore prepared project-local runtime artifacts.

Mutable databases, recordings, logs and validation output are excluded. A
runtime archive supplements the Git checkout; it is not a replacement OS image.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from import_runtime import relocate_environments

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "05_runtime"
MANIFEST = "bundle.manifest.json"
EXCLUDED = {
    "runtime-data",
    "cache",
    "validation",
    "exports",
    "state-transfer.json",
    "import-report.json",
}
SYSTEM_LINKS = {"/usr/bin/python3", "/usr/lib/python3.12/dist-packages/tensorrt"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def entries(root, *, include_git=False):
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(current)
        dirs[:] = [
            name
            for name in dirs
            if name not in {"__pycache__", ".pytest_cache", ".ruff_cache"}
            and (include_git or name != ".git")
            and not (base == root and name in EXCLUDED)
        ]
        for name in sorted([*dirs, *files]):
            path = base / name
            if base == root and (name in EXCLUDED or name == MANIFEST):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Invalid bundle path")
    path = root.joinpath(*relative.parts)
    if not path.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("Bundle parent escapes destination")
    return path


def inventory(root):
    files = {}
    for path in entries(root):
        name = str(path.relative_to(root))
        if path.is_symlink():
            target = str(path.readlink())
            if path.readlink().is_absolute() and target not in SYSTEM_LINKS:
                raise ValueError("Unexpected external symlink: " + name)
            if not path.readlink().is_absolute():
                lexical = Path(os.path.abspath(path.parent / path.readlink()))
                allowed_host_targets = {Path(p).resolve() for p in SYSTEM_LINKS}
                if not lexical.is_relative_to(root.resolve()) or (
                    not path.resolve().is_relative_to(root.resolve())
                    and path.resolve() not in allowed_host_targets
                ):
                    raise ValueError("Relative symlink escapes runtime: " + name)
            files[name] = {"kind": "symlink", "target": target}
        elif path.is_file():
            files[name] = {
                "kind": "file",
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    result = {
        "schema_version": 1,
        "installed_root": str(root.resolve()),
        "files": files,
        "file_count": len(files),
        "bytes": sum(x.get("bytes", 0) for x in files.values()),
        "excluded": sorted(EXCLUDED),
        "host_system_links": sorted(SYSTEM_LINKS),
    }
    (root / MANIFEST).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def verify(root):
    manifest = json.loads((root / MANIFEST).read_text())
    for name, entry in manifest["files"].items():
        path = safe_path(root, name)
        if entry["kind"] == "symlink":
            if not path.is_symlink() or str(path.readlink()) != entry["target"]:
                raise ValueError("Bundle link mismatch: " + name)
        elif path.is_symlink() or not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError("Bundle file mismatch: " + name)
    return manifest


def pack(root, output):
    manifest = verify(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w") as archive:
        archive.add(root / MANIFEST, arcname=MANIFEST, recursive=False)
        for path in entries(root, include_git=True):
            # Restore these explicit host links after data-filtered extraction.
            if path.is_symlink() and path.readlink().is_absolute():
                continue
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return {
        "archive": str(output),
        "sha256": digest(output),
        "artifact_bytes": manifest["bytes"],
        "archive_bytes": output.stat().st_size,
    }


def unpack(archive_path, expected_sha, destination):
    if digest(archive_path) != expected_sha:
        raise ValueError("Archive SHA256 mismatch")
    if destination.exists():
        raise ValueError("Destination already exists")
    destination.mkdir(parents=True)
    try:
        with tarfile.open(archive_path) as archive:
            archive.extractall(destination, filter="data")
        original = json.loads((destination / MANIFEST).read_text())
        for name, entry in original["files"].items():
            if entry["kind"] == "symlink" and Path(entry["target"]).is_absolute():
                if entry["target"] not in SYSTEM_LINKS:
                    raise ValueError("Unexpected host symlink")
                path = safe_path(destination, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(entry["target"])
        verify(destination)
        relocate_environments(destination, Path(original["installed_root"]))
        relocated = inventory(destination)
        return {
            "restored": str(destination),
            "files": relocated["file_count"],
            "source_archive_sha256": expected_sha,
        }
    except BaseException:
        # Only the newly created destination is owned by this operation.
        shutil.rmtree(destination)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("inventory")
    sub.add_parser("verify")
    packing = sub.add_parser("pack")
    packing.add_argument("--output", type=Path, required=True)
    restoring = sub.add_parser("unpack")
    restoring.add_argument("--archive", type=Path, required=True)
    restoring.add_argument("--sha256", required=True)
    restoring.add_argument("--destination", type=Path, default=RUNTIME)
    args = parser.parse_args()
    if args.action == "inventory":
        result = inventory(RUNTIME)
    elif args.action == "verify":
        result = verify(RUNTIME)
    elif args.action == "pack":
        if args.output.resolve().is_relative_to(
            RUNTIME
        ) and not args.output.resolve().is_relative_to(RUNTIME / "exports"):
            raise ValueError("Archives inside the runtime must be under exports/")
        result = pack(RUNTIME, args.output.resolve())
    else:
        result = unpack(args.archive.resolve(), args.sha256, args.destination.resolve())
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k != "files" or not isinstance(v, dict)
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
