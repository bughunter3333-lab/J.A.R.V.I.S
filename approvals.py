"""Human approval for risky actions.

A risky tool call (see config.NEEDS_APPROVAL) is not run. It is parked here and
Jarvis asks sir. Only sir's NEXT message can release it: "yes" runs it, "no"
cancels it, and anything else (or two minutes passing) drops it. The decision is
made by code reading sir's words, never by the model, so a prompt injection can't
approve its own action.
"""
import json
import re
from contextlib import closing
from datetime import datetime, timedelta

import config
import memory

YES = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|go ahead|do it|confirm(?:ed)?|approved?|"
                 r"please do|affirmative|proceed|go for it)\b[\s,.!]*(?:please|sir|jarvis|do it|go ahead)?[\s.!]*$", re.I)
NO = re.compile(r"^\s*(?!no (?:problem|worries|rush|stress)\b)"
                r"(?:no|nope|nah|don'?t|do not|cancel|stop|never ?mind|deny|denied|abort)\b", re.I)


def init_db() -> None:
    with closing(memory._connect()) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS approvals (id INTEGER PRIMARY KEY, session TEXT, "
                   "tool TEXT, args TEXT, summary TEXT, created TEXT, status TEXT DEFAULT 'pending')")


def needs_approval(tool: str, args: dict) -> bool:
    rule = config.NEEDS_APPROVAL.get(tool, False)
    if rule is True:
        return True
    if isinstance(rule, (set, list, tuple)):
        return str(args.get("action", "")).strip().lower() in rule
    return False


def describe(tool: str, args: dict) -> str:
    action = str(args.get("action", "")).lower()
    if tool == "pc_control" and action == "open_website":
        return f"open {args.get('url', 'a website')} in the browser"
    if tool == "pc_control" and action == "lock_screen":
        return "lock the screen"
    if tool == "clipboard":
        text = str(args.get("text", ""))
        return f'copy "{text[:60]}{"..." if len(text) > 60 else ""}" to the clipboard'
    if tool == "create_note":
        return f"save a note called \"{args.get('title', '')}\""
    return f"run {tool} with {json.dumps(args)[:80]}"


def request(session: str, tool: str, args: dict) -> dict:
    """Park an action. Asking twice for the same thing reuses the same request."""
    init_db()
    summary = describe(tool, args)
    args_json = json.dumps(args, sort_keys=True, default=str)
    with closing(memory._connect()) as db, db:
        db.execute("UPDATE approvals SET status='replaced' WHERE session=? AND status='pending' "
                   "AND NOT (tool=? AND args=?)", (session, tool, args_json))
        row = db.execute("SELECT id FROM approvals WHERE session=? AND status='pending' AND tool=? AND args=?",
                         (session, tool, args_json)).fetchone()
        if row:
            rid = row[0]
        else:
            rid = db.execute("INSERT INTO approvals (session, tool, args, summary, created) VALUES (?, ?, ?, ?, ?)",
                             (session, tool, args_json, summary, datetime.now().isoformat(timespec="seconds"))).lastrowid
    return {"id": rid, "summary": summary}


def pending(session: str, now: datetime | None = None) -> dict | None:
    """The action waiting for sir in this conversation, if it hasn't expired."""
    init_db()
    now = now or datetime.now()
    with closing(memory._connect()) as db, db:
        row = db.execute("SELECT id, tool, args, summary, created FROM approvals WHERE session=? AND status='pending' "
                         "ORDER BY id DESC LIMIT 1", (session,)).fetchone()
        if not row:
            return None
        rid, tool, args, summary, created = row
        if now - datetime.fromisoformat(created) > timedelta(seconds=config.APPROVAL_TIMEOUT):
            db.execute("UPDATE approvals SET status='expired' WHERE id=?", (rid,))
            return None
    return {"id": rid, "tool": tool, "args": json.loads(args), "summary": summary}


def expire_all() -> None:
    """Called when the server starts: nothing parked before a restart may still run."""
    init_db()
    with closing(memory._connect()) as db, db:
        db.execute("UPDATE approvals SET status='expired' WHERE status='pending'")


def settle(rid: int, status: str) -> None:
    with closing(memory._connect()) as db, db:
        db.execute("UPDATE approvals SET status=? WHERE id=?", (status, rid))


def decision(message: str) -> str | None:
    """'approve', 'deny', or None if the message isn't a yes/no answer."""
    if len(message.split()) > 8:
        return None
    if YES.match(message):
        return "approve"
    if NO.match(message):
        return "deny"
    return None
