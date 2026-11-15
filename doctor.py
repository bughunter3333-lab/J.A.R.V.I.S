"""Health check: finds the usual reasons Jarvis won't answer or won't talk.

    python doctor.py

Idea from OpenJarvis's `jarvis doctor` (Apache-2.0, https://github.com/open-jarvis/OpenJarvis).
Each check returns a status ("ok", "warn" or "fail"), what it found, and the exact fix.
The same checks back GET /health, which the HUD uses to explain an offline core.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import config

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL


@dataclass
class Check:
    name: str
    status: str  # "ok", "warn" or "fail"
    detail: str
    fix: str = ""


def _ollama_models() -> list[str] | None:
    """Names of installed models, or None if Ollama isn't answering."""
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=3) as res:
            return [m["name"] for m in json.load(res).get("models", [])]
    except Exception:
        return None


def check_python() -> Check:
    v = sys.version_info
    if v >= (3, 10):
        return Check("Python", "ok", f"{v.major}.{v.minor}.{v.micro}")
    return Check("Python", "fail", f"{v.major}.{v.minor} is too old", "Install Python 3.11.")


def check_ollama(models: list[str] | None) -> Check:
    if models is None:
        return Check("Ollama", "fail", f"not answering at {OLLAMA_URL}",
                     "Start it: open the Ollama app, or run `ollama serve` in another window.")
    return Check("Ollama", "ok", f"running, {len(models)} model(s) installed")


def check_model(models: list[str] | None) -> Check:
    if models is None:
        return Check("Model", "fail", "can't check, Ollama isn't running", "Fix Ollama first.")
    wanted = config.MODEL if ":" in config.MODEL else config.MODEL + ":latest"
    if wanted in models:
        return Check("Model", "ok", f"{config.MODEL} installed")
    return Check("Model", "fail", f"{config.MODEL} not found (Ollama sees: {', '.join(models) or 'nothing'})",
                 f"Run `ollama pull {config.MODEL}`, or if you moved it, run the model-folder check below.")


def _manifest_count(models_dir: Path) -> int:
    manifests = models_dir / "manifests"
    return sum(1 for p in manifests.rglob("*") if p.is_file()) if manifests.is_dir() else 0


def check_model_folder() -> Check:
    """Catches the 'moved the models but Ollama looks in the old place' problem."""
    default_dir = Path.home() / ".ollama" / "models"
    custom = os.environ.get("OLLAMA_MODELS")
    if not custom:
        n = _manifest_count(default_dir)
        return Check("Model folder", "ok" if n else "warn",
                     f"default ({default_dir}), {n} model file(s)",
                     "" if n else f"Pull a model: `ollama pull {config.MODEL}`")
    custom_dir = Path(custom)
    here, there = _manifest_count(custom_dir), _manifest_count(default_dir)
    if here:
        return Check("Model folder", "ok", f"OLLAMA_MODELS={custom_dir}, {here} model file(s)")
    if there:
        return Check("Model folder", "fail",
                     f"OLLAMA_MODELS points to {custom_dir} (empty) but models are still in {default_dir}",
                     f'Move them: robocopy "{default_dir}" "{custom_dir}" /E /MOVE, then restart Ollama.')
    return Check("Model folder", "warn", f"OLLAMA_MODELS={custom_dir} has no models",
                 f"Run `ollama pull {config.MODEL}`.")


def check_gpu() -> Check:
    if not shutil.which("nvidia-smi"):
        return Check("GPU", "warn", "nvidia-smi not found, the model may run on the CPU (slow)")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        name, total, used = [x.strip() for x in out.split(",")]
        free_gb = (int(total) - int(used)) / 1024
        detail = f"{name}, {free_gb:.1f} GB free of {int(total) / 1024:.1f} GB"
        # qwen3:8b needs roughly 5.6 GB when loaded; only warn if it isn't already loaded.
        if free_gb < 1.0:
            return Check("GPU", "warn", detail + " (very little free; is another model loaded?)",
                         "Close other GPU apps, or run `ollama ps` to see what's loaded.")
        return Check("GPU", "ok", detail)
    except Exception as exc:
        return Check("GPU", "warn", f"couldn't read nvidia-smi: {exc}")


def check_packages() -> Check:
    needed = {"fastapi": "fastapi", "uvicorn": "uvicorn", "ollama": "ollama", "kokoro": "kokoro",
              "soundfile": "soundfile", "sounddevice": "sounddevice", "RealtimeSTT": "RealtimeSTT"}
    missing = [pip for mod, pip in needed.items() if importlib.util.find_spec(mod) is None]
    if missing:
        return Check("Packages", "fail", "missing: " + ", ".join(missing),
                     "Run `pip install -r requirements.txt` inside the .venv.")
    return Check("Packages", "ok", "all installed")


def check_voice_model() -> Check:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    cached = (hub / "models--hexgrad--Kokoro-82M").is_dir()
    if cached:
        return Check("Voice model", "ok", "Kokoro-82M downloaded")
    return Check("Voice model", "warn", "Kokoro-82M not downloaded yet (about 330 MB)",
                 'Download it now so his first reply isn\'t silent: python voice.py "Good evening, sir."')


def check_data_dir() -> Check:
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        probe = config.DATA_DIR / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return Check("Data folder", "ok", str(config.DATA_DIR))
    except Exception as exc:
        return Check("Data folder", "fail", f"can't write to {config.DATA_DIR}: {exc}",
                     "Check the folder isn't read-only or open in another program.")


def run_checks(quick: bool = False) -> list[Check]:
    """quick=True skips the slower checks (GPU, packages) for the /health endpoint."""
    models = _ollama_models()
    checks = [check_ollama(models), check_model(models), check_model_folder(), check_data_dir(),
              check_voice_model()]
    if not quick:
        checks = [check_python()] + checks + [check_gpu(), check_packages()]
    return checks


def summary(checks: list[Check]) -> dict:
    worst = "fail" if any(c.status == "fail" for c in checks) else \
            "warn" if any(c.status == "warn" for c in checks) else "ok"
    return {"status": worst, "checks": [asdict(c) for c in checks]}


def main() -> int:
    marks = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
    checks = run_checks()
    print("JARVIS-X doctor\n")
    for c in checks:
        print(f"{marks[c.status]} {c.name:<13} {c.detail}")
        if c.fix and c.status != "ok":
            print(f"{'':20}fix: {c.fix}")
    result = summary(checks)["status"]
    print("\nAll systems go, sir." if result == "ok" else
          "\nSome warnings, but Jarvis should run." if result == "warn" else
          "\nJarvis can't run until the [FAIL] lines are fixed.")
    return 1 if result == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
