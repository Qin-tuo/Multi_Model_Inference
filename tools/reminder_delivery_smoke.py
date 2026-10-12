"""Real scheduled text delivery across console processes; synthetic reminders."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sqlalchemy import select
from app.reminders import ReminderService, _NOTIFICATIONS
from tools.project import stamp, write_json


def main():
    cases = []
    with tempfile.TemporaryDirectory(prefix="assistant-notifications-") as temporary:
        data = Path(temporary)
        database = data / "reminders.sqlite"
        with ReminderService(database) as service:
            due = datetime.now(timezone.utc) + timedelta(seconds=.1)
            service.create("owner", "合成测试提醒：喝水", due, request_id="owner")
            service.create("guest", "访客测试提醒：休息", due, request_id="guest")
            service.create("owner", "已过期测试提醒", due - timedelta(hours=2), request_id="expired")
            cancelled = service.create("owner", "已取消测试提醒", due, request_id="cancelled")
            service.cancel("owner", cancelled["reminder_id"], request_id="cancel")
            service.set_muted("owner", True)

        def run(name, owner, seconds):
            command = [sys.executable, str(ROOT / "tools/project.py"), "notifications", "--data-dir", str(data),
                       "--user", owner, "--seconds", str(seconds)]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=20)
            delivered = [line.removeprefix("提醒：") for line in result.stdout.splitlines() if line.startswith("提醒：")]
            row = {"name": name, "exit_code": result.returncode, "delivered": delivered}
            if result.returncode: row["error"] = result.stderr[-1200:]
            cases.append(row)
            return row

        muted = run("muted_process", "owner", .4)
        with ReminderService(database) as service: service.set_muted("owner", False)
        first = run("owner_delivery_process", "owner", .8)
        repeated = run("owner_restart_process", "owner", .4)
        guest = run("guest_delivery_process", "guest", .8)
        with ReminderService(database) as service:
            reminders = service.list_reminders("owner", include_inactive=True)
            with service.engine.connect() as connection:
                notices = [dict(row) for row in connection.execute(select(_NOTIFICATIONS)).mappings()]
    checks = {
        "all_commands_succeeded": all(row["exit_code"] == 0 for row in cases),
        "mute_persists_across_processes": muted["delivered"] == [],
        "owner_received_exactly_one_notice": first["delivered"] == ["合成测试提醒：喝水"],
        "completed_notice_not_repeated_after_restart": repeated["delivered"] == [],
        "guest_received_only_guest_notice": guest["delivered"] == ["访客测试提醒：休息"],
        "expired_and_cancelled_reminders_not_delivered": {r["status"] for r in reminders} == {"delivered", "expired", "cancelled"},
        "text_delivery_not_labelled_acoustic_playback": all(row["status"] == "displayed" and row["output_channel"] == "text"
            and row["played_utc"] is None and row["delivered_utc"] is not None for row in notices),
    }
    report = {"created_at": stamp(), "status": "PASS" if all(checks.values()) else "FAIL",
              "scope": "real APScheduler/SQLite and four separate text-output processes; no model, speaker, TTS quality or acoustic timing evidence",
              "checks": checks, "cases": cases, "notification_count": len(notices)}
    write_json(ROOT / "benchmarks/results/reminder-delivery-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
