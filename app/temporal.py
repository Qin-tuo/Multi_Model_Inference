"""Resolve Chinese reminder times with pinned JioNLP and Python datetime.

The model never performs duration arithmetic. Vague or multiple time entities
remain unresolved rather than acquiring an invented execution time.
"""
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import math
import re
import threading
from zoneinfo import ZoneInfo

_LOCK = threading.RLock()
_JIO = None
ISO = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})")


@dataclass(frozen=True)
class ReminderTime:
    due: datetime | None
    phrase: str = ""
    delay_seconds: int | None = None
    reason: str | None = None

    def arguments(self):
        if self.due is None: raise ValueError("Reminder time is unresolved")
        if self.delay_seconds is not None: return {"delay_seconds": self.delay_seconds, "due_at": None}
        return {"delay_seconds": None, "due_at": self.due.isoformat()}


def _parser():
    global _JIO
    if _JIO is None:
        # The package prints a promotional banner on import. Keep that out of
        # the assistant console; import errors and stderr remain visible.
        with redirect_stdout(io.StringIO()):
            import jionlp
        _JIO = jionlp
    return _JIO


def resolve_reminder_time(text, anchor, timezone_name="Asia/Shanghai"):
    local = anchor.astimezone(ZoneInfo(timezone_name))
    text = re.split(r"内容(?:是|为|[:：])", text, maxsplit=1)[0]
    explicit = list(ISO.finditer(text))
    if explicit:
        if len(explicit) != 1: return ReminderTime(None, reason="multiple_times")
        try: due = datetime.fromisoformat(explicit[0].group().replace("Z", "+00:00"))
        except ValueError: return ReminderTime(None, reason="invalid_time")
        return ReminderTime(due if due > anchor else None, explicit[0].group(), reason=None if due > anchor else "past_time")
    with _LOCK:
        jio = _parser()
        try:
            prefix = re.split(r"提醒(?:我|我们)", text, maxsplit=1)[0]
            entities = jio.ner.extract_time(prefix, time_base=local.replace(tzinfo=None), with_parsing=True, ret_future=False)
            if not entities:
                entities = jio.ner.extract_time(text, time_base=local.replace(tzinfo=None), with_parsing=True, ret_future=False)
        except (ValueError, TypeError): return ReminderTime(None, reason="invalid_time")
        if len(entities) != 1:
            return ReminderTime(None, reason="multiple_times" if entities else "missing_time")
        entity = entities[0]
        phrase = entity["text"]
        # A full-string duration parse handles Chinese numerals and compounds;
        # JioNLP's time-point intervals deliberately retain coarse precision.
        relative = (re.search(r"(?:之|以)?后$", phrase) or re.match(r"^(?:再)?(?:过|等)", phrase)
                    or re.search(r"(?:再)?(?:过|等)\s*" + re.escape(phrase), prefix))
        if relative:
            duration = re.sub(r"^(?:再)?(?:过|等)\s*", "", re.sub(r"(?:之|以)?后$", "", phrase))
            try: delta = jio.parse_time(duration, time_base=local.replace(tzinfo=None), time_type="time_delta", strict=True)
            except (ValueError, TypeError): delta = None
            if delta and delta.get("type") == "time_delta" and isinstance(delta.get("time"), dict):
                values = delta["time"]
                units = {"second":1, "minute":60, "hour":3600, "day":86400, "week":604800}
                precise = (delta.get("definition") == "accurate" or
                           ("半" in duration and not re.search(r"几|多|余|左右|大约|近|约", duration)))
                if precise and values and all(key in units and type(value) in (int, float) and math.isfinite(value) for key,value in values.items()):
                    seconds = sum(value * units[key] for key,value in values.items())
                    if float(seconds).is_integer() and 1 <= seconds <= 31536000:
                        return ReminderTime(anchor + timedelta(seconds=seconds), phrase, int(seconds))
                return ReminderTime(None, phrase, reason="ambiguous_duration")
        detail = entity.get("detail", {})
        if (detail.get("type") != "time_point" or detail.get("definition") != "accurate"
                or not re.search(r"点|时|[0-9][:：][0-9]", phrase)):
            return ReminderTime(None, phrase, reason="missing_clock_time")
        try: due = datetime.fromisoformat(detail["time"][0]).replace(tzinfo=ZoneInfo(timezone_name))
        except (ValueError, TypeError, KeyError, IndexError): return ReminderTime(None, phrase, reason="invalid_time")
        return ReminderTime(due if due > anchor else None, phrase, reason=None if due > anchor else "past_time")
