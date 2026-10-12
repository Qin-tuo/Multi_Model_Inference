"""Run persistent text notifications independently of model services."""
import argparse
import asyncio
import json
from pathlib import Path

from app.notifications import NotificationPump, TextNotificationSink
from app.reminders import ReminderService
from tools.project import DATA, ROOT


async def run(args):
    config = json.loads((ROOT / "configs/runtime.yaml").read_text())["reminders"]
    options = {key: config[key] for key in ("notification_ttl_seconds", "max_delivery_attempts", "retry_delay_seconds") if key in config}
    with ReminderService(args.data_dir / "reminders.sqlite", **options) as service:
        async def emit(text): print("提醒：" + text, flush=True)
        pump = NotificationPump(service, args.user, TextNotificationSink(service, emit), occupied=lambda: False)
        task = asyncio.create_task(pump.run())
        try:
            if args.seconds:
                try: await asyncio.wait_for(asyncio.shield(task), args.seconds)
                except TimeoutError: pass
            else:
                await task
        finally:
            await pump.stop()
            if not task.done(): await task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="owner")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--seconds", type=float, default=0, help="Optional bounded run; zero waits until stopped")
    args = parser.parse_args(argv)
    if args.seconds < 0: parser.error("seconds must be nonnegative")
    try: asyncio.run(run(args))
    except KeyboardInterrupt: pass
    return 0


if __name__ == "__main__": raise SystemExit(main())
