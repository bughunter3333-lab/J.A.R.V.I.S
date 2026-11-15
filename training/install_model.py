"""Load the fine-tuned GGUF into Ollama as a new model (default name: jarvis-x).

    python training\\install_model.py
    python training\\install_model.py --gguf path\\to\\model.gguf --name jarvis-x

The chat template is copied from the base qwen3 model already in Ollama, so tool calling
works exactly as before; only the weights change. Nothing is switched over until you
change MODEL in config.py, so the original qwen3:8b stays available.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_gguf() -> Path | None:
    found = sorted((HERE / "output").rglob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def build_modelfile(base_modelfile: str, gguf: Path) -> str:
    """Swap the base model's FROM line for our GGUF, keep its TEMPLATE / PARAMETERs / LICENSE."""
    lines = [ln for ln in base_modelfile.splitlines() if not ln.startswith("#")]
    text = "\n".join(lines)
    if not re.search(r"^FROM ", text, re.M):
        raise ValueError("The base modelfile has no FROM line.")
    return re.sub(r"^FROM .*$", f"FROM {gguf.resolve().as_posix()}", text, count=1, flags=re.M)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", type=Path, default=None)
    parser.add_argument("--name", default="jarvis-x")
    parser.add_argument("--template-from", default="qwen3:8b", help="Ollama model whose chat template to reuse")
    args = parser.parse_args()
    gguf = args.gguf or find_gguf()
    if not gguf or not gguf.exists():
        print("No GGUF found. Train first (training\\train_lora.py) or pass --gguf.")
        return 1
    base = subprocess.run(["ollama", "show", args.template_from, "--modelfile"],
                          capture_output=True, text=True, encoding="utf-8")
    if base.returncode != 0:
        print(f"Couldn't read {args.template_from} from Ollama: {base.stderr.strip()}")
        return 1
    modelfile = HERE / "output" / "Modelfile"
    modelfile.parent.mkdir(parents=True, exist_ok=True)
    modelfile.write_text(build_modelfile(base.stdout, gguf), encoding="utf-8")
    done = subprocess.run(["ollama", "create", args.name, "-f", str(modelfile)])
    if done.returncode != 0:
        return done.returncode
    print(f"\nInstalled as '{args.name}'. Compare before switching:")
    print(f"  python evals\\compare.py --models qwen3:8b {args.name}")
    print(f"If it wins, set MODEL = \"{args.name}\" in config.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
