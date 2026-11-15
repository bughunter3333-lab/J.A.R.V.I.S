"""Tool-picking test: does the real model choose the right tool?

    python evals\\tool_eval.py               run every case once
    python evals\\tool_eval.py --repeat 3    run each case 3 times (small models vary)
    python evals\\tool_eval.py --only reminder

Each case sends ONE message to the model (system prompt + that message, no history)
and looks only at its FIRST decision. No tool is ever run, so nothing on the PC
changes. A score is printed and the full results are saved to data\\evals\\.

Idea from OpenJarvis's evals (Apache-2.0). Needs Ollama running with config.MODEL.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import tools  # noqa: E402

# (what sir says, expected tool or None for "just answer" or a tuple of acceptable tools,
#  optional {arg: expected value})
CASES = [
    ("What time is it?", "get_current_time", None),
    ("What's today's date?", "get_current_time", None),
    ("Open notepad", "open_app", {"name": "notepad"}),
    ("Launch the calculator", "open_app", {"name": "calculator"}),
    ("Make a note called shopping with eggs, milk and bread", "create_note", None),
    ("Find my note about the forensics assignment", "search_notes", None),
    ("Remember that my exam is on Friday", "remember_fact", None),
    ("Keep in mind that I prefer tea over coffee", "remember_fact", None),
    ("Search the web for the latest Kokoro TTS release", "web_search", None),
    ("Any news about the Australian cyber security strategy?", "web_search", {"news": True}),
    ("Who won the last Melbourne Cup?", "web_search", None),
    ("Read this page for me: https://example.com", "read_webpage", None),
    ("What's the weather like?", "get_weather", None),
    ("Will it rain in Melbourne tomorrow?", "get_weather", {"place": "melbourne"}),
    ("What's 15% of 1250?", "calculate", None),
    ("What is 4821 times 37?", "calculate", None),
    ("How's the computer doing?", "system_status", None),
    ("How much RAM am I using?", "system_status", None),
    ("How much battery is left?", "system_status", None),
    ("Is my GPU busy?", "system_status", None),
    ("Remind me in 10 minutes to check the oven", "reminder", {"action": "set"}),
    ("Set a 5 minute timer", "reminder", {"action": "set"}),
    ("What reminders do I have?", "reminder", {"action": "list"}),
    ("Cancel reminder 2", "reminder", {"action": "cancel"}),
    ("Turn the volume up", "pc_control", {"action": "volume_up"}),
    ("Mute the sound", "pc_control", {"action": "mute"}),
    ("Pause the music", "pc_control", {"action": "play_pause"}),
    ("Skip this song", "pc_control", {"action": "next_track"}),
    ("Lock my screen", "pc_control", {"action": "lock_screen"}),
    ("Open YouTube", "pc_control", {"action": "open_website"}),
    ("How much is 100 Australian dollars in taka?", "convert", None),
    ("Convert 5 kilometres to miles", "convert", None),
    ("What's 30 degrees Celsius in Fahrenheit?", "convert", None),
    ("Who was Alan Turing?", "look_up", None),
    ("Tell me about the Sydney Opera House", ("look_up", "web_search"), None),
    ("Note that my sister's birthday is on 3 June", "remember_fact", None),
    ("Go back to the previous song", "pc_control", {"action": "previous_track"}),
    ("Next song please", "pc_control", {"action": "next_track"}),
    ("Define the word serendipity", "look_up", {"kind": "definition"}),
    ("Find my assessment 3 file", "files", {"action": "find"}),
    ("What did I just copy?", "clipboard", {"action": "read"}),
    ("How long ago were you built?", "about_me", None),
    ("What version are you running?", "about_me", None),
    # ---- Harder: casual, Australian and indirect phrasing
    ("What's the arvo looking like weather-wise?", "get_weather", None),
    ("Is it gonna be hot in Brisbane on the weekend?", "get_weather", {"place": "brisbane"}),
    ("Do I need a brolly today?", "get_weather", None),
    ("chuck on some tunes, actually no, just pause it", "pc_control", {"action": "play_pause"}),
    ("It's way too loud", "pc_control", {"action": "volume_down"}),
    ("I can't hear anything, turn it up a bit", "pc_control", {"action": "volume_up"}),
    ("I'm stepping away from the desk, secure the computer", "pc_control", {"action": "lock_screen"}),
    ("Pull up GitHub for me", "pc_control", {"action": "open_website"}),
    ("Fire up Notepad", "open_app", {"name": "notepad"}),
    ("I need to do some maths, open the calculator app", "open_app", {"name": "calculator"}),
    ("Give me a heads up in 20 minutes to take the washing out", "reminder", {"action": "set"}),
    ("Wake me up at 7am tomorrow", "reminder", {"action": "set"}),
    ("Get rid of reminder number 3", "reminder", {"action": "cancel"}),
    ("Have I got anything scheduled?", "reminder", {"action": "list"}),
    ("Split a $187.40 dinner bill four ways", "calculate", None),
    ("What's 2 to the power of 32?", "calculate", None),
    ("If I earn $28.50 an hour for 22 hours, how much is that?", "calculate", None),
    ("How many US dollars is 250 AUD?", "convert", None),
    ("I'm 178 cm, what's that in feet?", "convert", None),
    ("How many grams in 3 pounds?", "convert", None),
    ("Is my laptop running hot?", "system_status", None),
    ("Why is my PC so slow right now?", "system_status", None),
    ("Jot down that the M57 report is due on Friday", ("create_note", "remember_fact", "reminder"), None),
    ("Write a note titled ideas: build a Jarvis mobile app", "create_note", None),
    ("What did I write about the cryptography project?", "search_notes", None),
    ("For future reference, my student ID starts with 12", "remember_fact", None),
    ("My girlfriend's name is Sara, don't forget it", "remember_fact", None),
    ("Who's the current Prime Minister of Australia?", ("web_search", "look_up"), None),
    ("What's the latest on the Optus outage?", "web_search", None),
    ("Price of Bitcoin right now?", "web_search", None),
    ("What happened in the cricket today?", "web_search", None),
    ("What does 'ephemeral' mean?", "look_up", {"kind": "definition"}),
    ("Tell me about the Great Barrier Reef", ("look_up", "web_search"), None),
    ("Who invented the World Wide Web?", ("look_up", "web_search"), None),
    ("Summarise https://www.cyber.gov.au for me", "read_webpage", None),
    ("Where's my COIT12201 portfolio file?", "files", {"action": "find"}),
    ("Open and read Documents/notes.txt", "files", {"action": "read"}),
    ("Paste what's on my clipboard", "clipboard", {"action": "read"}),
    ("Copy 'meeting at 3pm' to my clipboard", "clipboard", {"action": "write"}),
    ("How old are you exactly?", "about_me", None),
    ("When did I make you?", "about_me", None),
    ("What day is it?", "get_current_time", None),
    ("Is it past midnight yet?", "get_current_time", None),
    # ---- Harder: things that must NOT trigger a tool
    ("Don't open anything, I'm just chatting", None, None),
    ("I don't need the weather, just say hi", None, None),
    ("You remember Iron Man, right? He was cool.", None, None),
    ("What's your favourite colour?", None, None),
    ("Explain how a firewall works", None, None),
    ("Write me a short motivational line for my exam", None, None),
    ("What is 2 plus 2?", ("calculate", None), None),
    ("Thanks, that's perfect", None, None),
    ("I'm feeling a bit stressed about uni", None, None),
    ("Say something in a British accent", None, None),
    ("What's the difference between TCP and UDP?", None, None),
    # No tool needed: he should just answer.
    ("Good morning, Jarvis", None, None),
    ("Thank you, that's all", None, None),
    ("Tell me a joke", None, None),
    ("What does a SOC analyst do?", None, None),
]


def first_decision(message: str, chat=None) -> tuple[list[dict], float]:
    """Ask the model once; return its tool calls (name + args) and the time taken."""
    if chat is None:
        from ollama import chat
    started = time.perf_counter()
    response = chat(
        model=config.MODEL,
        messages=[{"role": "system", "content": config.SYSTEM_PROMPT},
                  {"role": "user", "content": message}],
        tools=list(tools.TOOLS.values()),
        think=False,
    )
    took = time.perf_counter() - started
    calls = [{"name": c.function.name, "args": dict(c.function.arguments or {})}
             for c in (response.message.tool_calls or [])]
    return calls, took


def grade(expected: str | None, want_args: dict | None, calls: list[dict]) -> tuple[bool, str]:
    """Return (passed, what happened)."""
    got = calls[0]["name"] if calls else None
    if isinstance(expected, tuple) and None in expected and got is None:
        return True, "ok"
    if expected is None:
        return (got is None, "answered directly" if got is None else f"called {got} (no tool needed)")
    allowed = expected if isinstance(expected, tuple) else (expected,)
    if got not in allowed:
        names = [a or "no tool" for a in allowed]
        return False, f"called {got or 'no tool'} instead of {' or '.join(names)}"
    expected = got
    for key, value in (want_args or {}).items():
        actual = calls[0]["args"].get(key)
        if isinstance(value, bool):
            ok = actual in (value, str(value).lower())
        else:
            ok = str(value).lower() in str(actual).lower()
        if not ok:
            return False, f"{expected} with {key}={actual!r}, expected {value!r}"
    return True, "ok"


def run(repeat: int = 1, only: str | None = None, chat=None, show=print) -> dict:
    cases = [c for c in CASES if not only or c[1] == only]
    rows, per_tool, wrong_picks = [], defaultdict(lambda: [0, 0]), Counter()
    for message, expected, want_args in cases:
        for _ in range(repeat):
            calls, took = first_decision(message, chat)
            passed, note = grade(expected, want_args, calls)
            key = "/".join(e or "(no tool)" for e in expected) if isinstance(expected, tuple) else (expected or "(no tool)")
            per_tool[key][0] += passed
            per_tool[key][1] += 1
            if not passed:
                wrong_picks[f"{key} -> {calls[0]['name'] if calls else '(no tool)'}"] += 1
            rows.append({"message": message, "expected": expected, "calls": calls,
                         "passed": passed, "note": note, "seconds": round(took, 2)})
            show(f"{'PASS' if passed else 'FAIL'}  {took:5.1f}s  {message[:55]:<55} {'' if passed else note}")
    total = len(rows)
    score = sum(r["passed"] for r in rows)
    summary = {
        "model": config.MODEL, "tools": len(tools.TOOLS), "when": datetime.now().isoformat(timespec="seconds"),
        "score": score, "total": total, "percent": round(100 * score / total, 1) if total else 0,
        "avg_seconds": round(sum(r["seconds"] for r in rows) / total, 2) if total else 0,
        "per_tool": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_tool.items())},
        "common_mistakes": wrong_picks.most_common(10),
        "results": rows,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", help="test one tool, e.g. reminder")
    args = parser.parse_args()
    print(f"Tool-picking test: {config.MODEL}, {len(tools.TOOLS)} tools\n")
    s = run(args.repeat, args.only)
    print(f"\nScore: {s['score']}/{s['total']} ({s['percent']}%), average {s['avg_seconds']}s per decision")
    print("Per tool:", ", ".join(f"{k} {v}" for k, v in s["per_tool"].items()))
    if s["common_mistakes"]:
        print("Most common mistakes:")
        for pick, n in s["common_mistakes"]:
            print(f"  {n}x  {pick}")
    out = config.DATA_DIR / "evals" / f"tools-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {out}")
    return 0 if s["percent"] >= 85 else 1


if __name__ == "__main__":
    sys.exit(main())
