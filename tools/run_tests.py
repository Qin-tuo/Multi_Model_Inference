"""Record framework/plugin test evidence without equating fixtures to model tests."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.cases.append({"id": test.id(), "status": "PASS"})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.cases.append({"id": test.id(), "status": "SKIP", "reason": reason})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.cases.append({"id": test.id(), "status": "FAIL"})

    def addError(self, test, err):
        super().addError(test, err)
        self.cases.append({"id": test.id(), "status": "ERROR"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    versions = {}
    for package in ("pipecat-ai", "mem0ai", "APScheduler", "SQLAlchemy", "qdrant-client", "onnxruntime"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    status = "PASS" if result.wasSuccessful() and result.testsRun and not result.skipped else "INCOMPLETE"
    if not result.wasSuccessful(): status = "FAIL"
    report = {
        "schema_version": 1, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status, "scope": "framework and plugin contract tests; no model or acoustic quality acceptance",
        "embedding_input": "deterministic fixture; not BGE semantic verification",
        "llm_input": "labelled Pipecat test processor; not real model inference",
        "storage": "real temporary Qdrant and SQLite", "scheduler": "real APScheduler",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "versions": versions, "cases": result.cases,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_state": "working tree; see checkpoint commit containing this report",
    }
    dest = ROOT / "benchmarks/results/framework-tests.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
