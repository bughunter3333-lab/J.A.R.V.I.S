"""The actions Jarvis can take. Security rules live here in code, not in the prompt.

- Apps: only those in config.ALLOWED_APPS can be opened.
- Files: notes are written only inside config.WORKSPACE, never overwritten, .txt only.
- Every call (allowed or refused) goes into the audit log via run_tool().
"""
import re
import subprocess
from datetime import datetime
from pathlib import Path

import config
import memory

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
    return "\n".join(hits[:5]) if hits else "No notes matched."


def remember_fact(fact: str) -> str:
    """Remember a lasting fact about sir for future conversations.

    Args:
        fact: The fact to remember, as one short sentence.
    """
    if not fact.strip() or len(fact) > 300:
        raise ToolRefused("A fact must be one short sentence.")
    memory.add_fact(fact)
    return "Remembered."


TOOLS = {
    fn.__name__: fn
    for fn in (get_current_time, open_app, create_note, search_notes, remember_fact)
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
    memory.log_tool(name, args, result, ok)
    return result
