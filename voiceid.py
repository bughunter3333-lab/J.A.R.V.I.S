"""Know sir's voice: speaker recognition, so Jarvis only obeys sir, not the TV or a guest.

    python voiceid.py --enroll    read 5 short sentences aloud; saves sir's voiceprint
    python voiceid.py --check     say something; prints how close it is to sir's voice
    python voiceid.py --forget    delete the voiceprint (voice ID off again)

How: Resemblyzer (a small neural network, runs offline on the CPU) turns a few seconds of
speech into a 256-number "voiceprint". Recordings of the same person give similar
voiceprints. Enrolling stores the average of sir's; afterwards every request heard by
the HUD's local microphone is compared with it, and one that doesn't match
(config.VOICE_ID_THRESHOLD) is ignored.

Limits, on purpose:
- Off until sir enrols (and config.VOICE_ID can switch it off).
- Too short to judge ("yes", "stop"; under config.VOICE_ID_MIN_SECONDS of speech): accepted,
  or one-word answers would stop working.
- It is a convenience filter against the TV and other people, not a lock: a good
  recording of sir's voice would get through.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import config

SAMPLE_RATE = 16000  # RealtimeSTT records 16 kHz mono float32, which Resemblyzer expects
_encoder = None


def voiceprint_path() -> Path:
    return config.DATA_DIR / "voiceprint.npy"


def enrolled() -> bool:
    return voiceprint_path().exists()


def active() -> bool:
    return config.VOICE_ID and enrolled()


def _get_encoder():
    global _encoder
    if _encoder is None:
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)  # webrtcvad's pkg_resources notice
        from resemblyzer import VoiceEncoder

        _encoder = VoiceEncoder("cpu", verbose=False)
    return _encoder


def embed(audio) -> np.ndarray | None:
    """Voiceprint of one recording, or None if there is too little speech to judge."""
    from resemblyzer import preprocess_wav

    wav = np.asarray(audio, dtype=np.float32).flatten()
    if wav.size == 0:
        return None
    if np.abs(wav).max() > 1.5:  # int16 samples given as numbers: scale to -1..1
        wav = wav / 32768.0
    wav = preprocess_wav(wav, source_sr=SAMPLE_RATE)  # trims silences, normalises volume
    if wav.size < config.VOICE_ID_MIN_SECONDS * SAMPLE_RATE:
        return None
    return _get_encoder().embed_utterance(wav)


def enroll(clips: list) -> list[float]:
    """Save sir's voiceprint from several recordings. Returns how close each clip is to the
    average (all should be high; a low one means that clip was noisy or someone else)."""
    prints = [p for p in (embed(c) for c in clips) if p is not None]
    if len(prints) < 3:
        raise ValueError("Not enough clear speech: read each sentence in full, close to the microphone.")
    mean = np.mean(prints, axis=0)
    mean /= np.linalg.norm(mean)
    voiceprint_path().parent.mkdir(parents=True, exist_ok=True)
    np.save(voiceprint_path(), mean)
    return [float(np.dot(p, mean)) for p in prints]


def forget() -> None:
    voiceprint_path().unlink(missing_ok=True)


def similarity(audio) -> float | None:
    """How close a recording is to sir's voiceprint (about -1..1), or None if it can't say."""
    if not enrolled():
        return None
    p = embed(audio)
    if p is None:
        return None
    return float(np.dot(p, np.load(voiceprint_path())))


def is_sir(audio) -> bool | None:
    """True/False, or None when it can't tell (not enrolled, too short, library missing)."""
    if not active():
        return None
    try:
        score = similarity(audio)
    except Exception:  # resemblyzer missing or broken: never lock sir out
        return None
    return None if score is None else score >= config.VOICE_ID_THRESHOLD


# ---------------------------------------------------------------- command line

SENTENCES = (
    "Jarvis, what's the weather looking like this afternoon?",
    "Remind me to call my sister when I get home tonight.",
    "Turn the music down a little and pause the video.",
    "Search the web for the latest news about cyber security.",
    "Good morning, Jarvis. Read me my reminders for today, please.",
)


def _record(seconds: float) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enroll", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--forget", action="store_true")
    args = parser.parse_args()
    if args.forget:
        forget()
        print("Voiceprint deleted. Jarvis listens to anyone again.")
        return 0
    if args.check:
        if not enrolled():
            print("No voiceprint yet: run python voiceid.py --enroll first.")
            return 1
        input("Press Enter, then say anything for 4 seconds...")
        score = similarity(_record(4))
        if score is None:
            print("Too little speech to judge. Try again, closer to the microphone.")
            return 1
        verdict = "sir" if score >= config.VOICE_ID_THRESHOLD else "NOT sir"
        print(f"Similarity {score:.2f} (threshold {config.VOICE_ID_THRESHOLD}): {verdict}")
        return 0
    print("Read each sentence aloud in your normal voice, at your usual distance from the mic.\n")
    clips = []
    for i, sentence in enumerate(SENTENCES, 1):
        input(f"{i}/{len(SENTENCES)}  Press Enter, then say:  \"{sentence}\"")
        clips.append(_record(5))
    try:
        scores = enroll(clips)
    except ValueError as exc:
        print(exc)
        return 1
    print("Saved. Consistency of each clip: " + ", ".join(f"{s:.2f}" for s in scores))
    print("Now try: python voiceid.py --check   (you, then someone else or the TV)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
