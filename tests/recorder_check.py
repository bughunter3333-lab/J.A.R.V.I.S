"""Checks the real listening pipeline (wake word -> voice detection -> transcription)
without a microphone, by feeding it a recording of Jarvis's own voice.

Run:  python tests/recorder_check.py
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
import talk  # noqa: E402
import voice  # noqa: E402

REQUEST = "What time is it right now?"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    wav = voice.save(f"Hey Jarvis. {REQUEST}", config.AUDIO_DIR / "wake_test.wav")
    print("step: wav saved", flush=True)
    detected = []
    recorder = talk.make_recorder(use_wake_word=True, use_microphone=False)
    print("step: recorder ready", flush=True)
    recorder.on_wakeword_detected = lambda: detected.append(time.time())
    result = {}
    worker = threading.Thread(target=lambda: result.update(text=recorder.text()), daemon=True)
    worker.start()
    time.sleep(1.0)
    recorder.feed_audio_file(str(wav))
    print("step: audio fed", flush=True)
    silence = np.zeros(1600, dtype=np.int16)  # 0.1 s of quiet at 16 kHz
    deadline = time.time() + 45
    while worker.is_alive() and time.time() < deadline:
        recorder.feed_audio(silence, 16000)  # keeps "time passing" so end of speech is noticed
        time.sleep(0.1)
    heard = result.get("text")
    print(f"Wake word detected: {bool(detected)}")
    print(f"Heard after wake word: {heard!r}")
    ok = bool(detected) and heard is not None and "time" in heard.lower()
    print("RECORDER OK" if ok else "RECORDER CHECK FAILED", flush=True)
    # Shutting down can block while the listener thread is still waiting for speech,
    # so give it a few seconds and then exit regardless: the result is already printed.
    closer = threading.Thread(target=recorder.shutdown, daemon=True)
    closer.start()
    closer.join(5)
    return 0 if ok else 1


if __name__ == "__main__":
    import os

    code = main()
    sys.stdout.flush()
    os._exit(code)
