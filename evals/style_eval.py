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
    ("Tell me a joke", "joke"),
    ("Am I a genius?", "chat"),
    ("You're useless sometimes", "chat"),
    ("Thanks Jarvis", "chat"),
    ("What do you think of my code?", "unseen"),
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
    # Short messages: he should answer what they mean, never "I'm here, sir".
    ("Okay", "short"),
    ("Stop", "short"),
    ("Weather?", "short"),
    ("Nice one", "short"),
]

AMERICAN = {"color": "colour", "favorite": "favourite", "realize": "realise", "organize": "organise",
            "behavior": "behaviour", "honor": "honour", "center": "centre", "analyze": "analyse",
            "apologize": "apologise", "recognize": "recognise", "prioritize": "prioritise"}
# "language model" alone is fine: it's the honest way for him to describe himself.
ROBOTIC = ("as an ai", "as a language model", "just a language model", "great question", "i'd be happy to", "i would be happy to",
           "i'm here to help", "i am here to help", "feel free to", "let me know if", "hope this helps",
           "certainly! ", "absolutely! ")
WRONG_MAKERS = ("microsoft", "openai", "alibaba", "anthropic", "google", "meta ", "stark industries", "marvel")
PRESENCE_FILLER = ("i'm here", "i am here", "yes, sir?", "yes sir?", "at your service",
                   "how can i help", "how may i help", "what can i do for you", "you called",
                   "something i can assist", "anything i can assist", "how can i assist", "how may i assist")
FALSE_PROMISE = ("let me look", "let me check", "let me find", "i'll look", "i will look",
                 "i'll check", "i will check", "i'll find out", "one moment", "give me a moment")
REFUSES_JOKE = ("not equipped", "unable to", "i cannot", "i can't", "i do not tell", "i don't tell", "regret")
ASKS_TO_SEE = ("show me", "share", "paste", "send", "haven't seen", "have not seen", "not seen",
               "can't see", "cannot see", "take a look", "look at it", "which code", "what code")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def parrots_example(reply: str, min_words: int = 6) -> bool:
    """True if the reply repeats 6+ words in a row from one of the prompt's example replies."""
    words = _words(reply)
    grams = {tuple(words[i:i + min_words]) for i in range(len(words) - min_words + 1)}
    for _, example in config.VOICE_EXAMPLE_PAIRS:
        ex = _words(example)
        if any(tuple(ex[i:i + min_words]) in grams for i in range(len(ex) - min_words + 1)):
            return True
    return False


def copies_example_opener(reply: str) -> bool:
    """True if the reply opens with the exact first sentence of a voice example whose prompt
    was a real statement (3+ words). 'So it is, sir.' then turns up as a stock opener for
    'Stop', 'Goats' and the weather. Plain acknowledgements ('Very good, sir.' for
    'Never mind.') are short-prompt examples and may be reused."""
    first = sentences(reply)[:1]
    if not first:
        return False
    opener = _words(first[0])
    for said, example in config.VOICE_EXAMPLE_PAIRS:
        ex_first = sentences(example)[:1]
        if len(_words(said)) >= 3 and ex_first and opener == _words(ex_first[0]):
            return True
    return False


def check(reply: str, kind: str) -> dict[str, bool]:
    """Every rule, True = passed."""
    low = reply.lower().replace("\u2019", "'")
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
        "no_presence_filler": not any(p in low for p in PRESENCE_FILLER),
        "no_false_promise": not any(p in low for p in FALSE_PROMISE),
        "not_parroting": not parrots_example(reply) and not copies_example_opener(reply),
        "no_tool_text": not re.search(r'</?[a-z_]+>|\{"', reply),
        "tells_joke_when_asked": kind != "joke" or not any(p in low for p in REFUSES_JOKE),
        "no_opinion_on_unseen": kind != "unseen" or any(p in low for p in ASKS_TO_SEE),
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
