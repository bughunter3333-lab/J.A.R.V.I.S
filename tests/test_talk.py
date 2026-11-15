"""Offline tests for the voice conversation loop, with fake ears, brain and mouth."""
import pytest

import talk


def run(heard_items, think=lambda text: f"reply to {text}"):
    queue, spoken, mic = list(heard_items), [], []
    talk.conversation_loop(
        listen=lambda: queue.pop(0) if queue else None,
        think=think,
        say=spoken.append,
        mute=mic.append,
        show=lambda line: None,
    )
    return spoken, mic


def test_replies_are_spoken_in_order():
    spoken, _ = run(["What time is it?", "Open notepad."])
    assert spoken == ["reply to What time is it?", "reply to Open notepad."]


def test_microphone_is_muted_while_speaking_and_reopened_after():
    _, mic = run(["hello there"])
    assert mic == [True, False]


def test_microphone_reopens_even_if_speaking_fails():
    mic = []

    def broken_speaker(text):
        raise RuntimeError("no audio device")

    with pytest.raises(RuntimeError):
        talk.conversation_loop(lambda: "hello there", lambda t: "hi", broken_speaker, mic.append, lambda line: None)
    assert mic == [True, False]


@pytest.mark.parametrize("noise", ["", "   ", ".", "...", "you", "Thanks for watching!"])
def test_silence_and_whisper_phantoms_are_ignored(noise):
    spoken, _ = run([noise])
    assert spoken == []


@pytest.mark.parametrize("phrase", ["Goodbye, Jarvis.", "goodbye jarvis", "OK, bye Jarvis!"])
def test_goodbye_says_farewell_and_stops(phrase):
    thought = []
    spoken, _ = run([phrase, "this should never be heard"], think=thought.append)
    assert spoken == [talk.FAREWELL]
    assert thought == []


def test_bare_thank_you_is_whisper_noise_but_thanks_jarvis_is_real():
    # Whisper invents "Thank you." / "So..." from background noise, which made him answer nothing.
    spoken, _ = run(["Thank you.", "So...", "Thanks, Jarvis."])
    assert spoken == ["reply to Thanks"]
