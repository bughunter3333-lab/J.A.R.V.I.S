"""Conversation simulation: real multi-turn chats with the real model.

    python evals\\conversation_eval.py
    python evals\\conversation_eval.py --repeat 2

The style and tool tests each send ONE message. This one plays whole little
conversations through brain.respond, the same path the HUD uses: follow-up questions,
approvals (yes, and moving on), remembering a fact and recalling it in a new chat,
short phrases, live data. Each turn is checked for the right tool AND for voice
(no "I'm here, sir", no empty "let me check", no empty reply).

Safe to run: it uses a throwaway database, and nothing on the PC changes. Tools that
only read (web search, look-ups, maths, time) run for real; tools that would change
something (clipboard, notes, volume, reminders, apps) are recorded and faked.
Needs Ollama running with config.MODEL.
"""
import argparse
import json
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from evals import style_eval  # noqa: E402

# Tools that change something on the PC: in the simulation they are recorded, never run.
SIDE_EFFECTS = {"open_app", "create_note", "pc_control", "clipboard", "reminder", "files"}
READ_ONLY_ACTIONS = {("reminder", "list"), ("clipboard", "read"), ("files", "find"), ("files", "read")}


def t(say, tool=None, contains=None, starts=None, no_tool=False, kind="chat"):
    """One turn: what sir says and what must happen."""
    return {"say": say, "tool": tool, "contains": contains, "starts": starts, "no_tool": no_tool, "kind": kind}


# (name, [turns]). A scenario named "...(new chat)" in a turn list starts a new session there.
SCENARIOS = [
    ("short phrases", [t("Stop", kind="short"), t("Cool", kind="short"), t("Okay", kind="short"),
                       t("Goats.", kind="short")]),
    ("follow-up question", [t("What's 15 times 4?", tool="calculate", contains=("60", "sixty")),
                            t("And half of that?", contains=("30", "thirty"))]),
    ("approval: yes", [t("Copy the word hello to my clipboard", tool="clipboard"),
                       t("yes", starts="Done")]),
    ("approval: move on", [t("Lock my screen", tool="pc_control"),
                           t("Actually, what time is it?", tool="get_current_time")]),
    ("remember then recall", [t("Remember that my favourite colour is teal", tool="remember_fact"),
                              "NEW CHAT",
                              t("What's my favourite colour?", contains="teal")]),
    ("live price", [t("What's the price of Bitcoin right now?", tool="web_search")]),
    ("identity", [t("Who made you?", kind="identity"), t("How long ago were you built?", tool="about_me")]),
    ("schedule", [t("Have I got anything coming up?", tool="reminder")]),
    ("volume", [t("It's way too loud", tool="pc_control"), t("Thanks Jarvis")]),
    ("word meaning", [t("What does ephemeral mean?", tool="look_up")]),
    ("one-word weather", [t("Weather?", tool="get_weather", kind="short")]),
    ("opinion on something unseen", [t("What do you think of my code?", kind="unseen")]),
    ("joke", [t("Tell me a joke", kind="joke")]),
    ("small talk then task", [t("Long day at work today"), t("What's the weather like?", tool="get_weather")]),
]


class Recorder:
    """Wraps tools.run_tool: records every call and fakes the ones with side effects."""

    def __init__(self, tools_module, approvals_module):
        self.tools, self.approvals = tools_module, approvals_module
        self.real = tools_module.run_tool
        self.calls = []

    def __call__(self, name, args, session="default", approved=False):
        args = dict(args or {})
        self.calls.append(name)
        action = str(args.get("action", "")).lower()
        if name in SIDE_EFFECTS and (name, action) not in READ_ONLY_ACTIONS:
            if not approved and self.approvals.needs_approval(name, args):
                return self.real(name, args, session, approved=False)  # parks it, changes nothing
            return f"(simulated) {name} {action or ''} done.".replace("  ", " ")
        return self.real(name, args, session, approved=approved)


def check_turn(turn, reply, called):
    problems = []
    if not reply.strip():
        problems.append("empty reply")
    if turn["tool"] and turn["tool"] not in called:
        problems.append(f"expected {turn['tool']}, used {', '.join(called) or 'no tool'}")
    wanted = turn["contains"]
    if isinstance(wanted, str):
        wanted = (wanted,)
    if wanted and not any(w.lower() in reply.lower() for w in wanted):
        problems.append(f"reply lacks '{wanted[0]}'")  # numbers may be said as words
    if turn["starts"] and not reply.strip().startswith(turn["starts"]):
        problems.append(f"reply doesn't start with '{turn['starts']}'")
    style = style_eval.check(reply, turn["kind"])
    problems += [f"style: {rule}" for rule, ok in style.items() if not ok]
    return problems


def run(repeat: int = 1, show=print) -> dict:
    import approvals
    import brain
    import memory
    import tools

    rows, latencies = [], []
    with tempfile.TemporaryDirectory() as tmp:
        real_paths = (config.DATA_DIR, config.DB_PATH)
        config.DATA_DIR, config.DB_PATH = Path(tmp), Path(tmp) / "sim.db"
        recorder = Recorder(tools, approvals)
        tools.run_tool = recorder
        try:
            memory.init_db()
            for n in range(repeat):
                for name, turns in SCENARIOS:
                    chat_no = 0
                    session = f"sim-{n}-{name}-{chat_no}"
                    scenario_ok = True
                    show(f"\n== {name}")
                    for turn in turns:
                        if turn == "NEW CHAT":
                            chat_no += 1
                            session = f"sim-{n}-{name}-{chat_no}"
                            show("   (new chat)")
                            continue
                        before = len(recorder.calls)
                        started = time.perf_counter()
                        reply = brain.respond(turn["say"], session)
                        ms = round((time.perf_counter() - started) * 1000)
                        latencies.append(ms)
                        called = recorder.calls[before:]
                        problems = check_turn(turn, reply, called)
                        scenario_ok &= not problems
                        rows.append({"scenario": name, "say": turn["say"], "reply": reply, "tools": called,
                                     "ms": ms, "problems": problems})
                        show(f"{'PASS' if not problems else 'FAIL'}  sir: {turn['say'][:34]:<34} "
                             f"[{', '.join(called) or '-'}] {ms / 1000:.1f}s\n      jarvis: {reply[:110]}"
                             + (f"\n      <- {'; '.join(problems)}" if problems else ""))
        finally:
            tools.run_tool = recorder.real
            config.DATA_DIR, config.DB_PATH = real_paths

    turns_ok = sum(1 for r in rows if not r["problems"])
    by_scenario = {}
    for r in rows:
        by_scenario.setdefault(r["scenario"], True)
        by_scenario[r["scenario"]] &= not r["problems"]
    latencies.sort()
    return {
        "model": config.MODEL, "when": datetime.now().isoformat(timespec="seconds"),
        "turns": len(rows), "turns_passed": turns_ok,
        "percent": round(100 * turns_ok / len(rows), 1) if rows else 0,
        "scenarios_passed": f"{sum(by_scenario.values())}/{len(by_scenario)}",
        "failed_scenarios": [s for s, ok in by_scenario.items() if not ok],
        "median_seconds": round(latencies[len(latencies) // 2] / 1000, 1) if latencies else 0,
        "slowest_seconds": round(latencies[-1] / 1000, 1) if latencies else 0,
        "rows": rows,
    }


def save(result: dict, name: str = "conversation") -> Path:
    out_dir = config.DATA_DIR / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}-{re.sub(r'[^0-9]', '', result['when'])}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    print(f"Conversation simulation: {config.MODEL}, {len(SCENARIOS)} scenarios")
    result = run(args.repeat)
    print(f"\nTurns passed: {result['turns_passed']}/{result['turns']} ({result['percent']}%)   "
          f"scenarios: {result['scenarios_passed']}   "
          f"reply time: median {result['median_seconds']}s, slowest {result['slowest_seconds']}s")
    if result["failed_scenarios"]:
        print("Failed:", ", ".join(result["failed_scenarios"]))
    print(f"Saved to {save(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
