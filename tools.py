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
from datetime import datetime
from pathlib import Path

import config
import memory
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
    """Remember a lasting fact about sir for future conversations.

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


TOOLS = {
    fn.__name__: fn
    for fn in (
        get_current_time, open_app, create_note, search_notes, remember_fact,
        web_search, read_webpage, get_weather, calculate, system_status,
    )
}


def run_tool(name: str, args: dict) -> str:
    """Run one tool by name, enforce the registry, and write the audit log."""
    if name not in TOOLS:
        result, ok = f"Refused: there is no tool called '{name}'.", False
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
