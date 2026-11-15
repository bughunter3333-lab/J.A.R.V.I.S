"""Reminders and timers.

- Stored in SQLite, so they survive a restart (one missed while the server was off
  fires as soon as it starts again, marked as late).
- A background thread (started by main.py) checks every few seconds. When one is due
  it shows a Windows notification and queues it for the HUD, which says it out loud.
"""
import re
import subprocess
import sys
import threading
import time
from contextlib import closing
from datetime import datetime, timedelta

import memory

MAX_ACTIVE = 50
_UNITS = {"second": 1, "sec": 1, "s": 1, "minute": 60, "min": 60, "m": 60,
          "hour": 3600, "hr": 3600, "h": 3600, "day": 86400, "d": 86400}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10,
                 "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "half an": 0.5}


class BadTime(ValueError):
    pass


def init_db() -> None:
    with closing(memory._connect()) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY, message TEXT, due TEXT, "
            "created TEXT, status TEXT DEFAULT 'pending', announced INTEGER DEFAULT 0)"
        )


def _clock(text: str) -> tuple[int, int] | None:
    """'3pm', '3:30 pm', '15:30', '9', 'noon', 'midnight' -> (hour, minute)."""
    text = text.strip().lower().replace(".", "")
    if text in ("noon", "midday"):
        return 12, 0
    if text == "midnight":
        return 0, 0
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not m:
        return None
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def parse_when(when: str, now: datetime | None = None) -> datetime:
    """Understand the ways people say a time. Raises BadTime with a helpful message."""
    now = now or datetime.now()
    text = " ".join(when.strip().lower().split())
    text = re.sub(r"^(?:at|on|by|for)\s+", "", text)

    # "in 10 minutes", "in an hour and a half", "in 2h 30m", "10 minutes"
    rel = re.sub(r"^in\s+", "", text).replace("half an hour", "0.5 hour")
    word_nums = "|".join(sorted((w for w in _WORD_NUMBERS if " " not in w), key=len, reverse=True))
    pattern = (r"\b(?:(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|s|m|h|d)"
               rf"|({word_nums})\s+(seconds?|minutes?|mins?|hours?|days?))\b")
    matches = list(re.finditer(pattern, rel))
    leftover = re.sub(pattern, "", rel)
    leftover = re.sub(r"\b(and|a half|from now|later|time)\b", "", leftover).strip(" ,")
    if matches and not leftover:
        total = 0.0
        for m in matches:
            num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            n = float(num) if num[0].isdigit() else _WORD_NUMBERS[num]
            key = unit if unit in _UNITS else unit.rstrip("s")
            total += n * _UNITS[key]
        if "and a half" in rel:
            unit = (matches[-1].group(2) or matches[-1].group(4))
            total += 0.5 * _UNITS[unit if unit in _UNITS else unit.rstrip("s")]
        if total <= 0 or total > 365 * 86400:
            raise BadTime("That's outside the range I can set (up to a year ahead).")
        return now + timedelta(seconds=total)

    # ISO-style "2026-11-20 15:00" or "2026-11-20T15:00"
    try:
        iso = datetime.fromisoformat(text.replace(" at ", " "))
    except ValueError:
        iso = None
    if iso is not None:
        if iso <= now:
            _raise_past()
        return iso

    day = now.date()
    if text.startswith("tomorrow"):
        day, text = day + timedelta(days=1), text[len("tomorrow"):].strip()
    elif text.startswith("today") or text.startswith("tonight"):
        default_evening = text.startswith("tonight")
        text = text.split(None, 1)[1] if " " in text else ""
        if default_evening and not text:
            text = "8pm"
    else:
        for i, name in enumerate(_WEEKDAYS):
            if text.startswith(name) or text.startswith("next " + name):
                ahead = (i - now.weekday()) % 7 or 7
                day = day + timedelta(days=ahead)
                text = text.split(name, 1)[1].strip()
                break
    text = re.sub(r"^(?:at|morning|in the morning)\s*", "", text).strip()
    if text in ("morning", ""):
        text = text or ("9am" if day != now.date() else "")
    clock = _clock(text) if text else None
    if not clock:
        raise BadTime(f"I couldn't understand the time '{when}'. Try 'in 10 minutes', 'at 3pm' or 'tomorrow at 9am'.")
    hour, minute = clock
    # "at 3" with no am/pm: pick the next 3 o'clock that's still ahead
    if not re.search(r"am|pm|:\d\d|noon|midnight", text) and hour < 12:
        candidate = datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=minute)
        if candidate <= now and day == now.date():
            hour += 12
    due = datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=minute)
    if due <= now:
        if day == now.date():
            due += timedelta(days=1)  # "at 9am" said at 10am means tomorrow
        else:
            _raise_past()
    return due


def _raise_past():
    raise BadTime("That time has already passed.")


def add(message: str, when: str) -> tuple[int, datetime]:
    init_db()
    due = parse_when(when)
    with closing(memory._connect()) as db, db:
        active = db.execute("SELECT COUNT(*) FROM reminders WHERE status='pending'").fetchone()[0]
        if active >= MAX_ACTIVE:
            raise BadTime(f"There are already {MAX_ACTIVE} reminders waiting. Cancel some first.")
        cur = db.execute("INSERT INTO reminders (message, due, created) VALUES (?, ?, ?)",
                         (message.strip()[:300], due.isoformat(timespec="seconds"),
                          datetime.now().isoformat(timespec="seconds")))
        return cur.lastrowid, due


def pending() -> list[dict]:
    init_db()
    with closing(memory._connect()) as db:
        rows = db.execute("SELECT id, message, due FROM reminders WHERE status='pending' ORDER BY due").fetchall()
    return [{"id": i, "message": m, "due": d} for i, m, d in rows]


def cancel(reminder_id: int) -> bool:
    init_db()
    with closing(memory._connect()) as db, db:
        return db.execute("UPDATE reminders SET status='cancelled' WHERE id=? AND status='pending'",
                          (reminder_id,)).rowcount == 1


def fire_due(now: datetime | None = None, notify=None) -> list[dict]:
    """Mark due reminders as fired, notify for each, and return them."""
    init_db()
    now = now or datetime.now()
    notify = notify or windows_notification
    with closing(memory._connect()) as db, db:
        rows = db.execute("SELECT id, message, due FROM reminders WHERE status='pending' AND due <= ?",
                          (now.isoformat(timespec="seconds"),)).fetchall()
        for rid, _, _ in rows:
            db.execute("UPDATE reminders SET status='fired' WHERE id=?", (rid,))
    fired = []
    for rid, message, due in rows:
        late = (now - datetime.fromisoformat(due)).total_seconds() > 120
        item = {"id": rid, "message": message, "due": due, "late": late}
        fired.append(item)
        try:
            notify("Jarvis reminder" + (" (late)" if late else ""), message)
        except Exception as exc:
            print(f"Reminder notification failed: {exc}")
    return fired


def take_unannounced() -> list[dict]:
    """Fired reminders the HUD hasn't spoken yet (each is handed out once)."""
    init_db()
    with closing(memory._connect()) as db, db:
        rows = db.execute("SELECT id, message, due FROM reminders WHERE status='fired' AND announced=0").fetchall()
        db.execute("UPDATE reminders SET announced=1 WHERE status='fired' AND announced=0")
    return [{"id": i, "message": m, "due": d} for i, m, d in rows]


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def windows_notification(title: str, message: str) -> None:
    """Toast notification through PowerShell's built-in Windows APIs (no extra packages)."""
    if sys.platform != "win32":
        return
    script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$n = $t.GetElementsByTagName('text')
$n.Item(0).AppendChild($t.CreateTextNode({_ps_quote(title)})) > $null
$n.Item(1).AppendChild($t.CreateTextNode({_ps_quote(message)})) > $null
$app = '{{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}}\\WindowsPowerShell\\v1.0\\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($t))
"""
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                   capture_output=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def start_checker(interval: float = 5.0) -> threading.Thread:
    def loop():
        while True:
            try:
                fire_due()
            except Exception as exc:
                print(f"Reminder check failed: {exc}")
            time.sleep(interval)

    thread = threading.Thread(target=loop, daemon=True, name="reminders")
    thread.start()
    return thread


def describe(due: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now()
    when = due.strftime("%I:%M %p").lstrip("0")
    if due.date() == now.date():
        day = "today"
    elif due.date() == now.date() + timedelta(days=1):
        day = "tomorrow"
    else:
        day = due.strftime("%A %d %B")
    mins = round((due - now).total_seconds() / 60)
    ahead = f" (in {mins} minute{'s' * (mins != 1)})" if 0 < mins < 120 else ""
    return f"{day} at {when}{ahead}"
