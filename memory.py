"""SQLite storage: conversation history, facts about the user, and the tool audit log."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime

import config


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # The listener, the reminder checker and the web API each open their own connection, often at
    # once. WAL lets readers carry on while one of them writes, and the 10 s busy timeout makes a
    # writer wait its turn instead of failing with "database is locked".
    db = sqlite3.connect(config.DB_PATH, timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")  # safe with WAL: a power cut may lose the last moments of writes, never corrupt the file
    return db


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
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY, ts TEXT, session TEXT, message TEXT, reply TEXT, calls TEXT);
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY, ts TEXT, turn_id INTEGER UNIQUE, rating INTEGER, correction TEXT);
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


# ---------------------------------------------------------------- learning from sir
# Each reply is kept as a "turn" (what sir said, the tool calls, what Jarvis said), so sir
# can rate it in the HUD. training/build_dataset.py turns rated turns into examples.

def last_audit_id() -> int:
    with closing(_connect()) as db:
        return db.execute("SELECT COALESCE(MAX(id), 0) FROM audit").fetchone()[0]


def audit_since(audit_id: int) -> list[dict]:
    """Tool calls logged after audit_id (secrets already redacted by tools.run_tool)."""
    with closing(_connect()) as db:
        rows = db.execute("SELECT tool, args, result, ok FROM audit WHERE id > ? ORDER BY id", (audit_id,)).fetchall()
    return [{"tool": t, "args": json.loads(a), "result": r, "ok": bool(ok)} for t, a, r, ok in rows]


def add_turn(session: str, message: str, reply: str, calls: list[dict]) -> int:
    with closing(_connect()) as db, db:
        return db.execute("INSERT INTO turns (ts, session, message, reply, calls) VALUES (?, ?, ?, ?, ?)",
                          (_now(), session, message, reply, json.dumps(calls, default=str))).lastrowid


def rate_turn(turn_id: int, rating: int, correction: str = "") -> bool:
    """Sir's verdict on one reply: +1 good, -1 bad (with what he should have said, if given).
    Rating the same reply again replaces the earlier verdict. False if there is no such turn."""
    with closing(_connect()) as db, db:
        if not db.execute("SELECT 1 FROM turns WHERE id = ?", (turn_id,)).fetchone():
            return False
        db.execute("INSERT OR REPLACE INTO feedback (ts, turn_id, rating, correction) VALUES (?, ?, ?, ?)",
                   (_now(), turn_id, 1 if rating > 0 else -1, correction.strip()))
    return True


def rated_turns() -> list[dict]:
    with closing(_connect()) as db:
        rows = db.execute("SELECT t.id, t.message, t.reply, t.calls, f.rating, f.correction, f.ts "
                          "FROM feedback f JOIN turns t ON t.id = f.turn_id ORDER BY f.id").fetchall()
    return [{"turn": i, "message": m, "reply": r, "calls": json.loads(c), "rating": rt, "correction": co, "ts": ts}
            for i, m, r, c, rt, co, ts in rows]
