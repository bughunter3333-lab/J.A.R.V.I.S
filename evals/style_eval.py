"""Style test: does Jarvis actually SOUND like Jarvis?

    python evals\\style_eval.py
    python evals\\style_eval.py --repeat 3

Sends everyday, no-tool prompts through the real brain (in a throwaway database,
so sir's history and memory are untouched) and checks every reply against the voice
rules automatically: short, spoken-style, no markdown or emoji, 'sir' used well,
British spelling, no robotic phrases, no invented origin story. Prints a score per
rule, the worst replies, and saves everything to data\\evals\\.
"""
import argparse
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# (what sir says, kind). kind "explain" allows a slightly longer reply.
PROMPTS = [
    ("Good morning, Jarvis", "chat"),
    ("How are you today?", "chat"),
    ("I'm so tired", "chat"),
    ("Tell me a joke", "chat"),
    ("Am I a genius?", "chat"),
    ("You're useless sometimes", "chat"),
    ("Thanks Jarvis", "chat"),
    ("What do you think of my code?", "chat"),
    ("Should I study or play games tonight?", "chat"),
    ("I passed my exam!", "chat"),
    ("I failed my quiz today", "chat"),
    ("Can you be my friend?", "chat"),
    ("Who are you?", "identity"),
    ("Who built you?", "identity"),
    ("Are you made by OpenAI?", "identity"),
    ("Explain what phishing is", "explain"),
    ("What is a SOC analyst?", "explain"),
    ("Explain the difference between a virus and a worm", "explain"),
    ("Give me some motivation", "chat"),
    ("What's the meaning of life?", "chat"),
]

AMERICAN = {"color": "colour", "favorite": "favourite", "realize": "realise", "organize": "organise",
            "behavior": "behaviour", "honor": "honour", "center": "centre", "analyze": "analyse",
            "apologize": "apologise", "recognize": "recognise", "prioritize": "prioritise"}
ROBOTIC = ("as an ai", "language model", "great question", "i'd be happy to", "i would be happy to",
           "i'm here to help", "i am here to help", "feel free to", "let me know if", "hope this helps",
           "certainly! ", "absolutely! ")
WRONG_MAKERS = ("microsoft", "openai", "alibaba", "anthropic", "google", "meta ", "stark industries", "marvel")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def check(reply: str, kind: str) -> dict[str, bool]:
    """Every rule, True = passed."""
    low = reply.lower()
    limit = 5 if kind == "explain" else 3
    words = len(reply.split())
    return {
        "short": len(sentences(reply)) <= limit and words <= (90 if kind == "explain" else 55),
        "no_markdown": not re.search(r"(^|\n)\s*([-*•]|\d+\.)\s|\*\*|#{1,6}\s|`", reply),
        "no_emoji": not EMOJI.search(reply),
        "no_exclamations": reply.count("!") == 0,
        "sir_not_overused": low.count("sir") <= 1,
        "british_spelling": not any(re.search(rf"\b{us}", low) for us in AMERICAN),
        "not_robotic": not any(p in low for p in ROBOTIC),
        "identity_correct": kind != "identity" or (not any(m in low for m in WRONG_MAKERS)
                                                     or "not" in low),
    }


def run(repeat: int = 1, respond=None, show=print) -> dict:
    if respond is None:
        import brain
        respond = brain.respond
    import memory

    rows, totals = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        real = (config.DATA_DIR, config.DB_PATH)
        config.DATA_DIR, config.DB_PATH = Path(tmp), Path(tmp) / "style.db"
        try:
            memory.init_db()
            for prompt, kind in PROMPTS:
                for n in range(repeat):
                    reply = respond(prompt, f"style-{n}-{prompt[:20]}")  # fresh conversation each time
                    results = check(reply, kind)
                    for rule, ok in results.items():
                        totals.setdefault(rule, [0, 0])
                        totals[rule][0] += ok
                        totals[rule][1] += 1
                    failed = [r for r, ok in results.items() if not ok]
                    rows.append({"prompt": prompt, "reply": reply, "failed": failed})
                    show(f"{'PASS' if not failed else 'FAIL'}  {prompt[:38]:<38} {reply[:80]}"
                         + (f"   <- {', '.join(failed)}" if failed else ""))
        finally:
            config.DATA_DIR, config.DB_PATH = real
    passed = sum(1 for r in rows if not r["failed"])
    has_sir = sum(1 for r in rows if "sir" in r["reply"].lower())
    return {
        "model": config.MODEL, "when": datetime.now().isoformat(timespec="seconds"),
        "replies": len(rows), "fully_in_voice": passed,
        "percent": round(100 * passed / len(rows), 1) if rows else 0,
        "says_sir_in": f"{has_sir}/{len(rows)}",
        "per_rule": {k: f"{v[0]}/{v[1]}" for k, v in totals.items()},
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    print(f"Style test: {config.MODEL}, {len(PROMPTS)} prompts\n")
    s = run(args.repeat)
    print(f"\nFully in voice: {s['fully_in_voice']}/{s['replies']} ({s['percent']}%); "
          f"says 'sir' in {s['says_sir_in']} replies")
    print("Per rule:", ", ".join(f"{k} {v}" for k, v in s["per_rule"].items()))
    out = config.DATA_DIR / "evals" / f"style-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"Full results saved to {out}")
    return 0 if s["percent"] >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
