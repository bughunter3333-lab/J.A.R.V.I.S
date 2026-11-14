"""Jarvis's voice: text to speech with Kokoro, running on the CPU.

Try it:  python voice.py "Good evening, sir."
"""
import queue
import re
import sys
import threading
from pathlib import Path

import numpy as np

import config

SAMPLE_RATE = 24_000  # Kokoro always produces 24 kHz audio
_pipeline = None


def _get_pipeline():
    """Load the model once, on first use (the first call downloads it)."""
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline

        # "b" = British English. CPU keeps the graphics card free for the language model.
        _pipeline = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", device="cpu")
    return _pipeline


def _chunks(text: str):
    """Yield audio one sentence at a time, so speech can start before it is all ready."""
    text = text.strip()
    if not text:
        return
    one_sentence_per_line = re.sub(r"(?<=[.!?])\s+", "\n", text)
    for result in _get_pipeline()(one_sentence_per_line, voice=config.TTS_VOICE, speed=config.TTS_SPEED):
        if result.audio is not None:
            yield np.asarray(result.audio, dtype=np.float32)


def synthesize(text: str) -> np.ndarray:
    """Turn text into audio samples (mono float32 at 24 kHz)."""
    chunks = list(_chunks(text))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def save(text: str, path: Path) -> Path:
    """Speak into a .wav file instead of the speakers."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, synthesize(text), SAMPLE_RATE)
    return path


def speak(text: str) -> None:
    """Say the text out loud and wait until it has finished.

    A helper thread prepares the next sentence while the current one is playing,
    so he starts talking after the first sentence instead of after the whole reply.
    """
    import sounddevice as sd

    ready: queue.Queue = queue.Queue()

    def prepare():
        try:
            for chunk in _chunks(text):
                ready.put(chunk)
        finally:
            ready.put(None)  # always signal the end, even after an error

    threading.Thread(target=prepare, daemon=True).start()
    first = ready.get()
    if first is None:
        return
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        chunk = first
        while chunk is not None:
            stream.write(chunk.reshape(-1, 1))
            chunk = ready.get()


if __name__ == "__main__":
    speak(" ".join(sys.argv[1:]) or "Good evening, sir. All systems are online.")
