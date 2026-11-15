"""SQLite storage: conversation history, facts about the user, and the tool audit log."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime

import config


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(config.DB_PATH)


def init_db() -> None:
    with closing(_connect()) as db, db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, session TEXT, role TEXT, content TEXT, ts TEXT);
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY, fact TEXT UNIQUE, ts TEXT);
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY, ts TEXT, tool TEXT, args TEXT, result TEXT, ok INTEGER);
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY, ts TEXT, session TEXT, model TEXT,
                latency_ms INTEGER, rounds INTEGER, tools TEXT, ok INTEGER, error TEXT);
            """
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def add_message(session: str, role: str, content: str) -> None:
    with closing(_connect()) as db, db:
        db.execute(
            "INSERT INTO messages (session, role, content, ts) VALUES (?, ?, ?, ?)",
            (session, role, content, _now()),
        )


def recent_messages(session: str, limit: int = config.HISTORY_LIMIT) -> list[dict]:
    with closing(_connect()) as db:
        rows = db.execute(
            "SELECT role, content FROM messages WHERE session = ? ORDER BY id DESC LIMIT ?",
            (session, limit),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def add_fact(fact: str) -> None:
    with closing(_connect()) as db, db:
        db.execute("INSERT OR IGNORE INTO facts (fact, ts) VALUES (?, ?)", (fact.strip(), _now()))


def all_facts() -> list[str]:
    with closing(_connect()) as db:
        return [row[0] for row in db.execute("SELECT fact FROM facts ORDER BY id")]


def log_tool(tool: str, args: dict, result: str, ok: bool) -> None:
    """Audit log: every tool call is recorded, whether it worked or was refused."""
    with closing(_connect()) as db, db:
        db.execute(
            "INSERT INTO audit (ts, tool, args, result, ok) VALUES (?, ?, ?, ?, ?)",
            (_now(), tool, json.dumps(args, default=str), result[:500], int(ok)),
        )


def recent_audit(limit: int = 20) -> list[dict]:
    with closing(_connect()) as db:
        rows = db.execute(
            "SELECT ts, tool, args, result, ok FROM audit ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {"ts": ts, "tool": tool, "args": json.loads(args), "result": result, "ok": bool(ok)}
        for ts, tool, args, result, ok in rows
    ]


def add_trace(session: str, model: str, latency_ms: int, rounds: int, tools: list[str],
              ok: bool, error: str = "") -> None:
    """One row per request: how long it took, how many model rounds, which tools ran."""
    with closing(_connect()) as db, db:
        db.execute(
            "INSERT INTO traces (ts, session, model, latency_ms, rounds, tools, ok, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), session, model, int(latency_ms), rounds, json.dumps(tools), int(ok), error[:300]),
        )


def trace_stats(limit: int = 100) -> dict:
    """Summary of the most recent requests, for /stats and the HUD."""
    with closing(_connect()) as db:
        rows = db.execute(
            "SELECT latency_ms, ok, tools FROM traces ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    if not rows:
        return {"requests": 0}
    times = sorted(r[0] for r in rows)
    tool_count: dict[str, int] = {}
    for _, _, tools in rows:
        for name in json.loads(tools):
            tool_count[name] = tool_count.get(name, 0) + 1
    return {
        "requests": len(rows),
        "errors": sum(1 for r in rows if not r[1]),
        "latency_ms": {
            "median": times[len(times) // 2],
            "p95": times[min(len(times) - 1, int(len(times) * 0.95))],
            "last": rows[0][0],
        },
        "tools_used": tool_count,
    }


def clear_session(session: str) -> int:
    """Forget one conversation's history (facts and the audit log are kept)."""
    with closing(_connect()) as db, db:
        return db.execute("DELETE FROM messages WHERE session = ?", (session,)).rowcount
