"""Compare models on both tests: does the fine-tune actually beat the base model?

    python evals\\compare.py --models qwen3:8b jarvis-x
    python evals\\compare.py --models qwen3:8b jarvis-x --repeat 3

Runs the tool-picking test and the style test for each model and prints them side by
side, with speed. Keep a fine-tune only if it is better on BOTH: a model that sounds
more like Jarvis but picks the wrong tools more often is a step backwards.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from evals import style_eval, tool_eval  # noqa: E402


def score(model: str, repeat: int, quiet=True) -> dict:
    original = config.MODEL
    config.MODEL = model
    silent = (lambda *_: None) if quiet else print
    try:
        tools_result = tool_eval.run(repeat=repeat, show=silent)
        style_result = style_eval.run(repeat=repeat, show=silent)
    finally:
        config.MODEL = original
    return {"model": model, "tools": tools_result, "style": style_result}


def table(results: list[dict]) -> str:
    rows = [("", *[r["model"] for r in results]),
            ("Tool picking", *[f"{r['tools']['score']}/{r['tools']['total']} ({r['tools']['percent']}%)" for r in results]),
            ("Seconds per decision", *[str(r["tools"]["avg_seconds"]) for r in results]),
            ("Fully in voice", *[f"{r['style']['fully_in_voice']}/{r['style']['replies']} ({r['style']['percent']}%)"
                                 for r in results])]
    rules = results[0]["style"]["per_rule"].keys()
    rows += [(f"  {rule}", *[r["style"]["per_rule"][rule] for r in results]) for rule in rules]
    width = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(row, width)) for row in rows)


def verdict(base: dict, tuned: dict) -> str:
    t = tuned["tools"]["percent"] - base["tools"]["percent"]
    s = tuned["style"]["percent"] - base["style"]["percent"]
    if t >= 0 and s >= 0 and (t > 0 or s > 0):
        return f"{tuned['model']} is better (tools {t:+.1f} points, voice {s:+.1f}): worth switching."
    if t < 0:
        return f"{tuned['model']} picks tools worse ({t:+.1f} points): keep {base['model']}."
    return f"No clear gain (tools {t:+.1f}, voice {s:+.1f}): keep {base['model']}."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    results = []
    for model in args.models:
        print(f"Testing {model}...")
        results.append(score(model, args.repeat))
    print("\n" + table(results))
    if len(results) >= 2:
        print("\n" + verdict(results[0], results[-1]))
    out = config.DATA_DIR / "evals" / f"compare-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
