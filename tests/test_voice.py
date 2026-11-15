"""Offline tests for voice.py using a fake speech model and a fake sound card (no sound is made)."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import voice


@pytest.fixture
def fake_model(monkeypatch):
    """A stand-in for Kokoro: 100 samples per line of text it is given."""
    seen = []

    def pipeline(text, voice, speed):
        seen.append(text)
        for line in text.split("\n"):
            yield SimpleNamespace(audio=np.full(100, 0.5, dtype=np.float32))

    monkeypatch.setattr(voice, "_get_pipeline", lambda: pipeline)
    return seen


@pytest.fixture
def fake_speakers(monkeypatch):
    played = []

    class OutputStream:
        def __init__(self, **settings):
            played.append(("open", settings["samplerate"], settings["channels"]))

        def __enter__(self):
            return self

        def __exit__(self, *error):
            played.append(("close",))

        def write(self, data):
            played.append(("write", data.shape))

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=OutputStream))
    return played


def test_text_is_split_into_sentences(fake_model):
    audio = voice.synthesize("Good evening, sir. All systems are online! Shall I begin?")
    assert fake_model == ["Good evening, sir.\nAll systems are online!\nShall I begin?"]
    assert audio.shape == (300,) and audio.dtype == np.float32


def test_empty_text_makes_no_audio_and_never_loads_the_model(monkeypatch):
    monkeypatch.setattr(voice, "_get_pipeline", lambda: pytest.fail("model should not load"))
    assert voice.synthesize("   ").size == 0


def test_speak_plays_each_sentence_then_closes_the_stream(fake_model, fake_speakers):
    voice.speak("One. Two.")
    assert fake_speakers == [("open", 24000, 1), ("write", (100, 1)), ("write", (100, 1)), ("close",)]


def test_speak_with_nothing_to_say_never_opens_the_speakers(fake_model, fake_speakers):
    voice.speak("")
    assert fake_speakers == []


def test_save_writes_a_wav_file(fake_model, tmp_path):
    import soundfile as sf

    path = voice.save("Hello. There.", tmp_path / "sub" / "out.wav")
    data, rate = sf.read(path)
    assert rate == 24000 and len(data) == 200
