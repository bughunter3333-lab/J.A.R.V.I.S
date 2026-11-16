"""Voice ID logic, with a fake voiceprint model (resemblyzer isn't needed for these)."""
import numpy as np
import pytest

import config
import voiceid


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    # A "recording" here is just its voiceprint; [] means too little speech.
    monkeypatch.setattr(voiceid, "embed", lambda audio: None if len(audio) == 0
                        else np.asarray(audio, dtype=float) / np.linalg.norm(audio))


SIR = [1.0, 0.1, 0.0]
GUEST = [0.1, 1.0, 0.0]


def test_not_enrolled_means_no_verdict():
    assert not voiceid.enrolled() and voiceid.is_sir(SIR) is None


def test_enrol_then_tell_sir_from_a_guest():
    scores = voiceid.enroll([SIR, [1.0, 0.12, 0.0], [0.98, 0.1, 0.02]])
    assert min(scores) > 0.99
    assert voiceid.is_sir([1.0, 0.11, 0.0]) is True
    assert voiceid.is_sir(GUEST) is False
    assert voiceid.is_sir([]) is None            # too short: accepted by the listener


def test_switch_off_and_forget(monkeypatch):
    voiceid.enroll([SIR, SIR, SIR])
    monkeypatch.setattr(config, "VOICE_ID", False)
    assert voiceid.is_sir(GUEST) is None
    monkeypatch.setattr(config, "VOICE_ID", True)
    voiceid.forget()
    assert voiceid.is_sir(GUEST) is None


def test_enrolling_needs_enough_speech():
    with pytest.raises(ValueError, match="Not enough clear speech"):
        voiceid.enroll([SIR, [], []])


def test_a_broken_model_never_locks_sir_out(monkeypatch):
    voiceid.enroll([SIR, SIR, SIR])
    def boom(audio):
        raise ImportError("no resemblyzer")
    monkeypatch.setattr(voiceid, "embed", boom)
    assert voiceid.is_sir(GUEST) is None
