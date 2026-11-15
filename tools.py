"""The actions Jarvis can take. Security rules live here in code, not in the prompt.

- Apps: only those in config.ALLOWED_APPS can be opened.
- Files: notes are written only inside config.WORKSPACE, never overwritten, .txt only.
- Every call (allowed or refused) goes into the audit log via run_tool().
- Secrets and ID numbers are never written to notes or memory (see security.py).
- Note contents are handed to the model as untrusted data, never as instructions.
"""
import ast
import math
import operator
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config
import memory
import approvals
import pc
import reminders
import security
import web

MAX_NOTE_CHARS = 20_000


class ToolRefused(Exception):
    """Raised when a request breaks a security rule."""


def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%A %d %B %Y, %I:%M %p")


def open_app(name: str) -> str:
    """Open an application on the computer.

    Args:
        name: The app to open. One of: notepad, calculator, paint, file explorer.
    """
    key = name.strip().lower()
    if key not in config.ALLOWED_APPS:
        allowed = ", ".join(sorted(config.ALLOWED_APPS))
        raise ToolRefused(f"'{name}' is not on the allow-list. Allowed apps: {allowed}.")
    subprocess.Popen([config.ALLOWED_APPS[key]])
    return f"Opened {key}."


def _safe_note_path(title: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9 _-]", "", Path(title).stem).strip()[:60]
    if not stem:
        raise ToolRefused("The note title has no usable characters.")
    path = (config.WORKSPACE / f"{stem}.txt").resolve()
    if config.WORKSPACE.resolve() not in path.parents:
        raise ToolRefused("Notes can only be saved inside the Jarvis workspace folder.")
    return path


def create_note(title: str, content: str) -> str:
    """Save a new text note in the Jarvis workspace folder.

    Args:
        title: Short title for the note, used as the file name.
        content: The text to save in the note.
    """
    if len(content) > MAX_NOTE_CHARS:
        raise ToolRefused(f"Note is too long (limit {MAX_NOTE_CHARS} characters).")
    content = security.clean_for_storage("create_note", content)
    config.WORKSPACE.mkdir(parents=True, exist_ok=True)
    path = _safe_note_path(title)
    if path.exists():
        raise ToolRefused(f"A note called '{path.name}' already exists. Choose another title.")
    path.write_text(content, encoding="utf-8")
    return f"Saved note '{path.name}'."


def search_notes(query: str) -> str:
    """Search the saved notes for some text.

    Args:
        query: Word or phrase to look for in note titles and contents.
    """
    config.WORKSPACE.mkdir(parents=True, exist_ok=True)
    needle = query.strip().lower()
    hits = []
    for path in sorted(config.WORKSPACE.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if needle in path.stem.lower() or needle in text.lower():
            hits.append(f"{path.name}: {text[:200]}")
    if not hits:
        return "No notes matched."
    return security.as_untrusted("saved notes", "\n".join(hits[:5]))


def remember_fact(fact: str) -> str:
    """Save a lasting fact or preference about sir. Call this whenever sir says remember, keep in mind,
    don't forget or note that, or tells you something lasting about himself (likes, plans, people, dates).

    Args:
        fact: The fact to remember, as one short sentence.
    """
    if not fact.strip() or len(fact) > 300:
        raise ToolRefused("A fact must be one short sentence.")
    sensitive = [f for f in security.scan(fact, kinds=("secret", "pii")) if f.name != "email"]
    if sensitive:
        security.log("remember_fact", sensitive)
        raise ToolRefused(
            "That looks like a password, key or ID number. Those are never kept in memory."
        )
    memory.add_fact(fact)
    return "Remembered."


def web_search(query: str, news: bool = False) -> str:
    """Search the internet for current information, facts, or news.

    Args:
        query: What to search for, in a few words.
        news: True to search recent news articles instead of general web pages.
    """
    try:
        return web.search(query, news=bool(news))
    except web.Blocked as exc:
        raise ToolRefused(str(exc))


def read_webpage(url: str) -> str:
    """Open a public web page and read its text.

    Args:
        url: The full http or https address of the page.
    """
    try:
        return web.fetch(url)
    except web.Blocked as exc:
        raise ToolRefused(str(exc))


def get_weather(place: str = "") -> str:
    """Get the current weather and today's and tomorrow's forecast.

    Args:
        place: City or town name. Leave empty for sir's home city.
    """
    try:
        return web.weather(place)
    except web.Blocked as exc:
        raise ToolRefused(str(exc))


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "log": math.log, "log10": math.log10, "abs": abs, "round": round}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    """Walk the parsed expression and allow only numbers, maths operators and a few functions.
    Nothing is ever passed to eval(), so the model can't run code through this tool."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolRefused("That power is too large.")
        result = _OPS[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > 4000:
            raise ToolRefused("That number is too large.")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolRefused("Only numbers, + - * / % ** and sqrt, sin, cos, tan, log, abs, round are allowed.")


def calculate(expression: str) -> str:
    """Work out a maths expression exactly. Use this instead of doing arithmetic in your head.

    Args:
        expression: The sum to work out, e.g. "(1250 * 0.15) + 42" or "sqrt(2) * 10".
    """
    expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/").strip()
    if len(expr) > 200:
        raise ToolRefused("That expression is too long.")
    try:
        value = _eval(ast.parse(expr, mode="eval"))
    except ToolRefused:
        raise
    except ZeroDivisionError:
        raise ToolRefused("That divides by zero.")
    except Exception:
        raise ToolRefused("I couldn't read that as a maths expression.")
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return f"{expression.strip()} = {value:,.10g}" if isinstance(value, float) else f"{expression.strip()} = {value:,}"


def system_status() -> str:
    """Report how the computer is doing: CPU, memory, disk space, battery and graphics card."""
    import psutil

    parts = [f"CPU {psutil.cpu_percent(interval=0.5):.0f}%"]
    mem = psutil.virtual_memory()
    parts.append(f"memory {mem.used / 2**30:.1f} of {mem.total / 2**30:.1f} GB used")
    disk = psutil.disk_usage(str(config.BASE_DIR.anchor or "/"))
    parts.append(f"drive {config.BASE_DIR.anchor or '/'} {disk.free / 2**30:.0f} GB free")
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if battery:
        parts.append(f"battery {battery.percent:.0f}%{' and charging' if battery.power_plugged else ''}")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()[0]
            util, used, total, temp = [x.strip() for x in out.split(",")]
            parts.append(f"GPU {util}% busy, {int(used) / 1024:.1f} of {int(total) / 1024:.1f} GB memory, {temp}°C")
        except Exception:
            pass
    return ", ".join(parts) + "."


# ---------------------------------------------------------------- reminders

def reminder(action: str, message: str = "", when: str = "", number: int = 0) -> str:
    """Set, list or cancel reminders and timers. Sir gets a Windows notification and you say it aloud.

    Args:
        action: "set" to create one, "list" to see waiting ones, or "cancel" to remove one.
        message: For "set": what to remind sir about, e.g. "submit the forensics report".
        when: For "set": when, in sir's words, e.g. "in 10 minutes", "at 3pm", "tomorrow at 9am", "friday 2pm".
        number: For "cancel": the reminder number shown by "list".
    """
    action = action.strip().lower()
    if action in ("set", "add", "create", "timer"):
        if not message.strip():
            message = "Timer finished."
        try:
            rid, due = reminders.add(security.clean_for_storage("reminder", message), when)
        except reminders.BadTime as exc:
            raise ToolRefused(str(exc))
        return f"Reminder {rid} set for {reminders.describe(due)}: {message.strip()}"
    if action in ("list", "show"):
        waiting = reminders.pending()
        if not waiting:
            return "No reminders waiting."
        from datetime import datetime as _dt
        return "\n".join(f"{r['id']}. {reminders.describe(_dt.fromisoformat(r['due']))}: {r['message']}"
                         for r in waiting)
    if action in ("cancel", "delete", "remove"):
        if not number:
            raise ToolRefused("Which one? Give the reminder number from the list.")
        if reminders.cancel(int(number)):
            return f"Reminder {int(number)} cancelled."
        raise ToolRefused(f"There's no waiting reminder number {int(number)}.")
    raise ToolRefused('The action must be "set", "list" or "cancel".')


# ---------------------------------------------------------------- PC control

def pc_control(action: str, url: str = "") -> str:
    """Control the computer: volume, music playback, lock the screen, or open a website in the browser.

    Args:
        action: One of: volume_up (louder), volume_down (quieter), mute (mute or unmute),
            play_pause (play, pause or resume), next_track (skip this song, next song),
            previous_track (go back to the previous song, replay the last one),
            lock_screen (lock the computer), open_website (open a site in the browser).
        url: Only for open_website: the website address, e.g. "youtube.com".
    """
    try:
        return pc.control(action, url)
    except pc.NotAllowed as exc:
        raise ToolRefused(str(exc))


# ---------------------------------------------------------------- conversions

# Everything is converted through one base unit per kind.
_UNIT_TABLE = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1, "km": 1000, "in": 0.0254, "inch": 0.0254, "ft": 0.3048,
               "foot": 0.3048, "feet": 0.3048, "yd": 0.9144, "yard": 0.9144, "mi": 1609.344, "mile": 1609.344},
    "mass": {"mg": 1e-6, "g": 0.001, "kg": 1, "t": 1000, "tonne": 1000, "oz": 0.0283495, "ounce": 0.0283495,
             "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "stone": 6.35029},
    "volume": {"ml": 0.001, "l": 1, "litre": 1, "liter": 1, "cup": 0.25, "tsp": 0.00492892, "tbsp": 0.0147868,
               "floz": 0.0295735, "gal": 3.78541, "gallon": 3.78541, "pint": 0.473176},
    "speed": {"kmh": 1, "km/h": 1, "kph": 1, "mph": 1.609344, "m/s": 3.6, "mps": 3.6, "knot": 1.852, "knots": 1.852},
    "data": {"b": 1, "byte": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12, "kib": 1024, "mib": 1024**2,
             "gib": 1024**3, "tib": 1024**4},
    "time": {"s": 1, "sec": 1, "second": 1, "min": 60, "minute": 60, "h": 3600, "hr": 3600, "hour": 3600,
             "day": 86400, "week": 604800},
}
_TEMPS = {"c": "c", "celsius": "c", "°c": "c", "f": "f", "fahrenheit": "f", "°f": "f", "k": "k", "kelvin": "k"}


_ALIASES = {
    "kilometre": "km", "kilometer": "km", "centimetre": "cm", "centimeter": "cm", "millimetre": "mm",
    "millimeter": "mm", "metre": "m", "meter": "m", "inche": "in", "kilogram": "kg", "kilo": "kg",
    "gram": "g", "milligram": "mg", "millilitre": "ml", "milliliter": "ml", "teaspoon": "tsp",
    "tablespoon": "tbsp", "byte": "b", "kilobyte": "kb", "megabyte": "mb", "gigabyte": "gb", "terabyte": "tb",
    "kilometres per hour": "km/h", "kilometers per hour": "km/h", "miles per hour": "mph", "hr": "h",
}


def _unit_key(unit: str) -> str:
    """'Kilometres' -> 'km', 'lbs' -> 'lbs', 'Miles' -> 'mile', 'GB' -> 'gb'."""
    u = " ".join(unit.strip().lower().split())
    if u in _ALIASES:
        return _ALIASES[u]
    u = u.replace(" ", "")
    known = {k for t in _UNIT_TABLE.values() for k in t} | set(_TEMPS)
    if u in known:
        return u
    singular = u[:-1] if u.endswith("s") else u
    singular = _ALIASES.get(singular, singular)
    return singular if singular in known else u


def _convert_temp(value: float, a: str, b: str) -> float:
    c = value if a == "c" else (value - 32) * 5 / 9 if a == "f" else value - 273.15
    return c if b == "c" else c * 9 / 5 + 32 if b == "f" else c + 273.15


def convert(amount: float, from_unit: str, to_unit: str) -> str:
    """Convert money between currencies (live daily rates) or units like km/miles, kg/lb, °C/°F, GB/MB.

    Args:
        amount: The number to convert.
        from_unit: Currency code (AUD, BDT, USD, EUR...) or unit (km, mile, kg, lb, C, F, litre, GB...).
        to_unit: The currency code or unit to convert into.
    """
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ToolRefused("The amount must be a number.")
    a, b = from_unit.strip(), to_unit.strip()
    known_units = {k for t in _UNIT_TABLE.values() for k in t} | set(_TEMPS)
    if (len(a) == 3 and len(b) == 3 and a.isalpha() and b.isalpha()
            and _unit_key(a) not in known_units and _unit_key(b) not in known_units):
        try:
            rate, updated = web.exchange_rate(a, b)
        except web.Blocked as exc:
            raise ToolRefused(str(exc))
        return (f"{amount:,.2f} {a.upper()} = {amount * rate:,.2f} {b.upper()} "
                f"(rate {rate:,.4f}, updated {updated}, source open.er-api.com)")
    ka, kb = _unit_key(a), _unit_key(b)
    if ka in _TEMPS and kb in _TEMPS:
        return f"{amount:g}°{_TEMPS[ka].upper()} = {_convert_temp(amount, _TEMPS[ka], _TEMPS[kb]):,.1f}°{_TEMPS[kb].upper()}"
    for table in _UNIT_TABLE.values():
        if ka in table and kb in table:
            result = amount * table[ka] / table[kb]
            return f"{amount:g} {a} = {result:,.4g} {b}"
    raise ToolRefused(f"I can't convert {a} to {b}. They need to be two currencies or two units of the same kind.")


# ---------------------------------------------------------------- knowledge

def look_up(topic: str, kind: str = "wikipedia") -> str:
    """Look up a person, place, event, organisation or concept on Wikipedia, or a word in the dictionary.
    Use this for "who was / who is / what is / tell me about" questions instead of answering from memory.

    Args:
        topic: The subject or word to look up.
        kind: "wikipedia" for an encyclopedia summary, or "definition" for a word's meaning.
    """
    try:
        if kind.strip().lower() in ("definition", "define", "dictionary", "meaning"):
            return web.define(topic)
        return web.wikipedia(topic)
    except web.Blocked as exc:
        raise ToolRefused(str(exc))


# ---------------------------------------------------------------- files and clipboard

def files(action: str, query: str) -> str:
    """Find files by name, or read a document, inside the folders sir has allowed.

    Args:
        action: "find" to search file names, or "read" to open one file (txt, md, csv, pdf, docx and similar).
        query: For "find": part of the file name, e.g. "assessment". For "read": the full path from "find".
    """
    try:
        if action.strip().lower() in ("find", "search", "locate"):
            return pc.find_files(query)
        if action.strip().lower() in ("read", "open", "summarise", "summarize"):
            return pc.read_file(query)
    except pc.NotAllowed as exc:
        raise ToolRefused(str(exc))
    raise ToolRefused('The action must be "find" or "read".')


def clipboard(action: str, text: str = "") -> str:
    """Read what sir has copied, or put text on the clipboard so he can paste it.

    Args:
        action: "read" to see the clipboard, or "write" to copy text onto it.
        text: For "write": the text to copy.
    """
    try:
        if action.strip().lower() in ("read", "get", "paste"):
            return pc.read_clipboard()
        if action.strip().lower() in ("write", "copy", "set"):
            if not text:
                raise ToolRefused("There's nothing to copy.")
            return pc.write_clipboard(text)
    except pc.NotAllowed as exc:
        raise ToolRefused(str(exc))
    raise ToolRefused('The action must be "read" or "write".')


# ---------------------------------------------------------------- about Jarvis himself

STARTED_AT = datetime.now()  # when this server process started


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(config.BASE_DIR), *args], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _local(dt: datetime) -> datetime:
    """Convert to this PC's local time and drop the timezone, so all maths is like-for-like."""
    return dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt


def build_time() -> tuple[datetime | None, str]:
    """When Jarvis was built, and where that answer came from."""
    if getattr(config, "BUILT_AT", None):
        return datetime.fromisoformat(str(config.BUILT_AT)), "the BUILT_AT setting in config.py"
    candidates = []
    if sys.platform == "win32":  # on Windows, st_ctime is when the folder was created
        candidates.append((datetime.fromtimestamp(config.BASE_DIR.stat().st_ctime),
                           f"the date the {config.BASE_DIR} folder was created"))
    first = _git("rev-list", "--max-parents=0", "HEAD").splitlines()
    if first:
        stamp = _git("show", "-s", "--format=%aI", first[-1])
        if stamp:
            candidates.append((_local(datetime.fromisoformat(stamp)), "the first git commit"))
    if not candidates:
        return None, ""
    return min(candidates, key=lambda c: c[0])


def exact_duration(start: datetime, end: datetime) -> str:
    """'2 days, 3 hours, 12 minutes and 5 seconds' (worked out in code, not by the model)."""
    seconds = int((end - start).total_seconds())
    if seconds < 0:
        return "in the future (check the clock or the BUILT_AT setting)"
    parts = []
    for name, size in (("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)):
        value, seconds = divmod(seconds, size)
        if value or (name == "second" and not parts):
            parts.append(f"{value} {name}{'s' if value != 1 else ''}")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _stamp(dt: datetime) -> str:
    return dt.strftime("%A %d %B %Y at %I:%M:%S %p").replace(" 0", " ")


def about_me() -> str:
    """Facts about yourself: exactly when you were built and how long ago, your version, and how long you have been running."""
    now = datetime.now()
    lines = []
    built, source = build_time()
    if built:
        lines.append(f"Built on {_stamp(built)} (from {source}), which was exactly "
                     f"{exact_duration(built, now)} ago.")
    else:
        lines.append("I can't tell when I was built: set BUILT_AT in config.py.")
    head = _git("log", "-1", "--format=%h|%aI|%s")
    if head:
        sha, when, subject = head.split("|", 2)
        commits = _git("rev-list", "--count", "HEAD")
        lines.append(f"Current version: {sha}, updated {exact_duration(_local(datetime.fromisoformat(when)), now)} "
                     f"ago ({subject}); {commits} changes in total.")
    lines.append(f"Running since {_stamp(STARTED_AT)}, for {exact_duration(STARTED_AT, now)}.")
    lines.append(f"Mind: {config.MODEL} via Ollama. Voice: Kokoro {config.TTS_VOICE}. "
                 f"Tools: {len(TOOLS)}. Right now it is {_stamp(now)}.")
    return " ".join(lines)


TOOLS = {
    fn.__name__: fn
    for fn in (
        get_current_time, open_app, create_note, search_notes, remember_fact,
        web_search, read_webpage, get_weather, calculate, system_status,
        reminder, pc_control, convert, look_up, files, clipboard, about_me,
    )
}


def run_tool(name: str, args: dict, session: str = "default", approved: bool = False) -> str:
    """Run one tool by name, enforce the registry and approvals, and write the audit log.

    `approved` is only ever set by brain.py after sir himself said yes; the model
    can't pass it (it isn't a tool argument).
    """
    if name not in TOOLS:
        result, ok = f"Refused: there is no tool called '{name}'.", False
    elif not approved and approvals.needs_approval(name, args):
        req = approvals.request(session, name, args)
        result, ok = (f"WAITING FOR APPROVAL: I need sir's permission to {req['summary']}. "
                      "This has NOT been done yet. Ask sir whether to go ahead."), False
    else:
        try:
            result, ok = TOOLS[name](**args), True
        except ToolRefused as exc:
            result, ok = f"Refused: {exc}", False
        except Exception as exc:  # bad arguments from the model, OS errors, etc.
            result, ok = f"Failed: {type(exc).__name__}: {exc}", False
    # The audit log must not become a copy of the secrets the tools refused to store.
    safe_args = {k: security.redact(v) if isinstance(v, str) else v for k, v in args.items()}
    memory.log_tool(name, safe_args, security.redact(result), ok)
    return result
