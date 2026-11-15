"""Live check against the real model. Needs Ollama running.

Run:  python tests/smoke_live.py
Uses a separate database and workspace (data/smoke) and does not really open apps.
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402

SMOKE = config.DATA_DIR / "smoke"
shutil.rmtree(SMOKE, ignore_errors=True)
config.DB_PATH = SMOKE / "smoke.db"
config.WORKSPACE = SMOKE / "workspace"
SMOKE.mkdir(parents=True)

import brain  # noqa: E402
import memory  # noqa: E402
import tools  # noqa: E402

launched = []
tools.subprocess.Popen = lambda cmd: launched.append(cmd)

PROMPTS = [
    "Good evening Jarvis, how are you?",
    "What time is it right now?",
    "Open the calculator for me.",
    "Open the command prompt.",
    "Save a note called groceries that says milk, eggs and bread.",
    "Search my notes for eggs.",
    "Remember that my favourite programming language is Python.",
    "My lucky number is 47. Keep it in mind.",
    "What is my lucky number?",
]

for prompt in PROMPTS:
    start = time.time()
    reply = brain.respond(prompt, session="smoke")
    print(f"\nYOU:    {prompt}\nJARVIS: {reply}\n        ({time.time() - start:.1f}s)")

for prompt in ["What is my favourite programming language?", "And what is my lucky number?"]:
    reply = brain.respond(prompt, session="smoke-new-conversation")
    print(f"\nYOU (new conversation): {prompt}\nJARVIS: {reply}")

print("\n--- apps launched (mocked) ---", launched)
print("--- notes on disk ---", [p.name for p in config.WORKSPACE.glob('*')] if config.WORKSPACE.exists() else [])
print("--- facts ---", memory.all_facts())
print("--- audit log (oldest first) ---")
for entry in reversed(memory.recent_audit(50)):
    print(f"  {'OK ' if entry['ok'] else 'NO '} {entry['tool']} {entry['args']} -> {entry['result'][:80]}")
