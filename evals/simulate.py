"""Run every simulation in one go and print one summary to paste back.

    python evals\\simulate.py           style + tool picking + conversations
    python evals\\simulate.py --quick   conversations only (about 3 minutes)

Each part also prints its own details. The summary at the end, and the file it
is saved to (data\\evals\\simulation-<time>.txt), are what to share.
"""
import argparse
import io
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from evals import conversation_eval, style_eval, tool_eval  # noqa: E402


def failures(rows, key, limit=12):
    lines = []
    for r in rows:
        if r.get(key):
            what = r[key] if isinstance(r[key], str) else "; ".join(r[key])
            said = r.get("say") or r.get("prompt") or r.get("message", "")
            lines.append(f"  - {said[:40]!r}: {what[:120]}  | reply: {str(r.get('reply', ''))[:90]!r}")
    return lines[:limit] + ([f"  ... and {len(lines) - limit} more"] if len(lines) > limit else [])


def gpu_share() -> str:
    """'ollama ps' shows how much of the model sits on the GPU; anything on CPU makes replies slow."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return "ollama ps: not available"
    rows = [line for line in out.splitlines()[1:] if line.strip()]
    return "ollama ps: " + (" | ".join(" ".join(r.split()) for r in rows) or "no model loaded")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="conversations only")
    args = parser.parse_args()

    summary = [f"JARVIS-X simulation  {datetime.now():%Y-%m-%d %H:%M}  model {config.MODEL}"]

    print("\n######## Conversations ########")
    conv = conversation_eval.run()
    summary.append(f"Conversations: {conv['turns_passed']}/{conv['turns']} turns ({conv['percent']}%), "
                   f"scenarios {conv['scenarios_passed']}, reply time median {conv['median_seconds']}s, "
                   f"slowest {conv['slowest_seconds']}s")
    summary += failures(conv["rows"], "problems")

    if not args.quick:
        print("\n######## Style ########")
        style = style_eval.run()
        summary.append(f"Style: {style['fully_in_voice']}/{style['replies']} fully in voice ({style['percent']}%)")
        summary += failures(style["results"], "failed")

        print("\n######## Tool picking ########")
        with redirect_stdout(io.StringIO()):  # 100+ lines; the summary lists the misses
            tools_result = tool_eval.run()
        summary.append(f"Tools: {tools_result['score']}/{tools_result['total']} ({tools_result['percent']}%), "
                       f"avg {tools_result['avg_seconds']}s")
        misses = [r for r in tools_result["results"] if not r["passed"]]
        summary += [f"  - {r['message'][:45]!r}: {r['note'][:110]}" for r in misses[:15]]

    summary.append(gpu_share())
    text = "\n".join(summary)
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)
    out = config.DATA_DIR / "evals" / f"simulation-{datetime.now():%Y%m%d-%H%M%S}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Saved to {out}  (paste the block above to Claude)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
