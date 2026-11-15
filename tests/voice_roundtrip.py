"""Voice check with no microphone or speakers: text -> Kokoro -> .wav -> Whisper -> text.

Run:  python tests/voice_roundtrip.py
The first run downloads the voice and speech-recognition models.
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402
import voice  # noqa: E402

SENTENCE = "Good evening, sir. All systems are online and the kettle is on."


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


start = time.time()
wav = voice.save(SENTENCE, config.AUDIO_DIR / "roundtrip.wav")
first = time.time() - start

start = time.time()
audio = voice.synthesize(SENTENCE)
second = time.time() - start
seconds = audio.size / voice.SAMPLE_RATE
print(f"TTS: {seconds:.1f}s of audio. First call {first:.1f}s (includes loading), warm call {second:.1f}s.")
print(f"Saved: {wav}")

from faster_whisper import WhisperModel  # noqa: E402

start = time.time()
model = WhisperModel(config.STT_MODEL, device="cpu", compute_type="int8")
load = time.time() - start
start = time.time()
segments, _ = model.transcribe(str(wav), language="en")
heard = " ".join(segment.text.strip() for segment in segments)
print(f"STT: loaded in {load:.1f}s, transcribed in {time.time() - start:.1f}s.")
print(f"Said:  {SENTENCE}\nHeard: {heard}")

expected = words(SENTENCE)
matched = sum(1 for word in expected if word in words(heard))
score = matched / len(expected)
print(f"Word match: {matched}/{len(expected)} ({score:.0%})")
assert seconds > 2, "audio is too short to be real speech"
assert score >= 0.8, "transcription does not match what was said"
print("ROUNDTRIP OK")
