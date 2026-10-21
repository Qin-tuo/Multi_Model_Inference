# SPDX-License-Identifier: Apache-2.0
"""Relocate copied Python environments without modifying their source runtime."""

from pathlib import Path
from project_paths import ROOT, LEGACY_RUNTIME_PATHS


def normalize_runtime_layout(destination):
    """Upgrade an unpacked legacy bundle before rebasing its environment scripts."""
    for old, new in LEGACY_RUNTIME_PATHS.items():
        source, target = destination / old, destination / new
        if source.exists():
            if target.exists():
                raise ValueError("Both legacy and current runtime paths exist: " + new)
            source.rename(target)


def relocate_environments(destination, source):
    """Repair venv entry scripts; Python executables still use host Python 3.12."""
    changed = []
    for name in ("edge-assistant", "speech"):
        env = destination / "environments" / name
        original_env = source / "environments" / name
        physical_env = original_env.resolve()
        source_roots = {source, physical_env.parents[1]}
        aliases = [str(original_env), str(physical_env)]
        aliases += [
            str(base / old)
            for old, new in LEGACY_RUNTIME_PATHS.items()
            if new == "environments/" + name
            for base in (source, destination)
        ]
        if name == "speech":
            aliases += [str(ROOT / ".venv-speech")]
        for file in [env / "pyvenv.cfg", *list((env / "bin").iterdir())]:
            if file.is_symlink() or not file.is_file():
                continue
            try:
                text = file.read_text()
            except UnicodeDecodeError:
                continue
            updated = text
            for old in aliases:
                updated = updated.replace(old, str(env))
            if name == "edge-assistant":
                for old, new in LEGACY_RUNTIME_PATHS.items():
                    if new == "environments/" + name:
                        updated = updated.replace(
                            "(" + Path(old).name + ")", "(" + name + ")"
                        )
            if updated != text:
                file.write_text(updated)
                changed.append(str(file.relative_to(destination)))
        # Future editable installs must not retain an external source root.
        site = env / "lib/python3.12/site-packages"
        for file in [*site.glob("*.pth"), *site.glob("__editable__*.py")]:
            text = file.read_text()
            updated = text
            for root in source_roots:
                updated = updated.replace(str(root), str(destination))
            if updated != text:
                file.write_text(updated)
                changed.append(str(file.relative_to(destination)))
    return changed

