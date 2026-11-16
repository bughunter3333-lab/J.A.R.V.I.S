"""Offline tests for local listening, with a fake RealtimeSTT recorder (no microphone)."""
import json
import queue
import time

from fastapi.testclient import TestClient

import config
import listener as listener_module
from listener import Listener


class FakeRecorder:
    def __init__(self, phrases, **callbacks):
        self.phrases, self.cb = queue.Queue(), callbacks
        for p in phrases:
            self.phrases.put(p)
        self.mic_on, self.closed = True, False
        self.last_transcription_bytes = [0.0] * 16000  # RealtimeSTT keeps the last utterance's audio
        self.use_wake_words = True  # the real RealtimeSTT attribute conversation mode flips

    def text(self):
        phrase = self.phrases.get(timeout=5)
        if phrase is None:
            raise RuntimeError("stopped")
        self.cb["on_wakeword_detected"]()
        self.cb["on_recording_start"]()
        self.cb["on_recording_stop"]()
        return phrase

    def set_microphone(self, on):
        self.mic_on = on

    def abort(self):
        self.phrases.put(None)

    def shutdown(self):
        self.closed = True


def make_listener(phrases):
    made = {}

    def factory(use_wake_word=True, **callbacks):
        made["rec"] = FakeRecorder(phrases, **callbacks)
        made["wake_word"] = use_wake_word
        return made["rec"]

    return Listener(make_recorder=factory), made


def wait_for(listener, kind, after=0, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for e in listener.events_after(after, timeout=0.2):
            if e["type"] == kind:
                return e
    raise AssertionError(f"no {kind} event")


def test_heard_speech_becomes_an_event_and_noise_is_ignored():
    lst, made = make_listener(["thanks for watching", "what's the weather"])
    lst.start()
    heard = wait_for(lst, "heard")
    assert heard["text"] == "what's the weather"
    kinds = [e["type"] for e in lst.events_after(0, timeout=0)]
    assert kinds[:2] == ["loading", "waiting"] and "wake" in kinds and "transcribing" in kinds
    assert [e for e in lst.events_after(0, 0) if e["type"] == "heard"] == [heard]  # phantom phrase dropped
    lst.stop()
    assert made["rec"].closed and lst.state == "off"


def test_mute_switches_the_recorder_microphone():
    lst, made = make_listener([])
    lst.start(wake_word=False)
    wait_for(lst, "waiting")
    assert made["wake_word"] is False
    lst.mute(True)
    assert made["rec"].mic_on is False
    lst.mute(False)
    assert made["rec"].mic_on is True
    lst.stop()


def test_microphone_failure_is_reported():
    def broken(**kwargs):
        raise OSError("No input device")
    lst = Listener(make_recorder=broken)
    lst.start()
    err = wait_for(lst, "error")
    assert "No input device" in err["detail"] and lst.state == "error"


def test_start_twice_does_not_open_two_microphones():
    lst, made = make_listener([])
    lst.start()
    first = made["rec"] if wait_for(lst, "waiting") else None
    lst.start()
    time.sleep(0.2)
    assert made["rec"] is first
    lst.stop()


def test_sse_stream_ends_by_itself():
    lst, _ = make_listener([])
    started = time.monotonic()
    chunks = list(lst.sse(0, max_seconds=0.5))
    assert time.monotonic() - started < 3
    assert json.loads(chunks[0].split("data: ", 1)[1])["type"] == "state"


def test_endpoints(monkeypatch):
    import listener as listener_module
    import main
    lst, made = make_listener(["open notepad"])
    monkeypatch.setattr(listener_module, "listener", lst)
    client = TestClient(main.app)
    assert client.post("/listen/start", json={"wake_word": True}).json() == {"state": "loading"}
    wait_for(lst, "heard")
    assert client.post("/listen/mute", json={"muted": True}).json() == {"muted": True}
    assert made["rec"].mic_on is False
    assert client.post("/listen/stop").json() == {"state": "off"}



# ---------------------------------------------------------------- conversation mode

def wait_conversation_end(lst, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        found = [e for e in lst.events_after(0, 0) if e["type"] == "conversation" and not e["active"]]
        if found:
            return found[0]
        time.sleep(0.05)
    return None


def heard_texts(lst):
    return [e["text"] for e in lst.events_after(0, 0) if e["type"] == "heard"]


def test_conversation_starts_with_hey_jarvis_and_ends_with_thats_all():
    lst, made = make_listener(["what's the weather", "and tomorrow?", "That's all, thank you"])
    lst.start()
    end = wait_for(lst, "conversation", after=0)            # first: conversation started
    assert end["active"] is True
    done = wait_conversation_end(lst)
    assert done and done["reason"] == "phrase" and done["farewell"] == config.CONVERSATION_FAREWELL
    # the follow-up was passed on without a wake word; the goodbye was not sent to the brain
    assert heard_texts(lst) == ["what's the weather", "and tomorrow?"]
    assert made["rec"].use_wake_words is True and lst.conversation is False
    lst.stop()


def test_wake_word_is_off_during_a_conversation():
    lst, made = make_listener(["open notepad"])
    lst.start()
    wait_for(lst, "heard")
    assert lst.conversation is True and made["rec"].use_wake_words is False
    lst.stop()


def test_silence_ends_the_conversation(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_IDLE_TIMEOUT", 1)
    lst, made = make_listener(["hello"])
    lst.start()
    wait_for(lst, "heard")
    end = wait_conversation_end(lst, timeout=6)
    assert end and end["reason"] == "idle" and end["farewell"] == ""
    assert made["rec"].use_wake_words is True
    lst.stop()


def test_idle_clock_is_paused_while_he_speaks(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_IDLE_TIMEOUT", 1)
    lst, _ = make_listener(["tell me a long story"])
    lst.start()
    wait_for(lst, "heard")
    lst.mute(True)                       # Jarvis is talking
    time.sleep(2.5)
    assert lst.conversation is True      # still in the conversation
    lst.mute(False)
    lst.stop()


def test_timeout_zero_means_never(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_IDLE_TIMEOUT", 0)
    lst, _ = make_listener(["hi"])
    lst.start()
    wait_for(lst, "heard")
    time.sleep(1.5)
    assert lst.conversation is True
    lst.stop()


def test_open_mic_mode_has_no_conversations():
    lst, made = make_listener(["hello", "that's all"])
    lst.start(wake_word=False)
    time.sleep(1.0)
    assert lst.conversation is False
    assert not [e for e in lst.events_after(0, 0) if e["type"] == "conversation"]
    lst.stop()


import pytest  # noqa: E402


@pytest.mark.parametrize("said", ["That's all, thank you", "thank you, that's all", "Thanks Jarvis, that will be all.",
                                  "Goodbye Jarvis", "that's it for now, thanks", "we're done", "okay thank you that's all"])
def test_end_phrases(said):
    assert listener_module.is_conversation_end(said)


@pytest.mark.parametrize("said", ["that's all I wanted to ask about the weather", "thank you", "what's the weather",
                                  "is that all the reminders?", "say goodbye to my friend", "stop the music"])
def test_not_end_phrases(said):
    assert not listener_module.is_conversation_end(said)


# ---------------------------------------------------------------- wake word choice

def test_jarvis_uses_porcupine_builtin_keyword(monkeypatch):
    import talk
    monkeypatch.setattr(config, "WAKE_PHRASE", "jarvis")
    opts = talk.wake_backend_options()
    assert opts["wakeword_backend"] == "pvporcupine" and opts["wake_words"] == "jarvis"
    assert opts["wake_words_sensitivity"] == config.WAKE_SENSITIVITY
    assert "openwakeword_model_paths" not in opts


def test_hey_jarvis_still_available_with_openwakeword(monkeypatch):
    import talk
    monkeypatch.setattr(config, "WAKE_PHRASE", "hey jarvis")
    monkeypatch.setattr(talk, "wake_word_model_path", lambda: "/models/hey_jarvis.onnx")
    opts = talk.wake_backend_options()
    assert opts["wakeword_backend"] == "oww" and opts["openwakeword_model_paths"] == "/models/hey_jarvis.onnx"


def test_waiting_event_tells_the_hud_the_wake_phrase():
    lst, _ = make_listener([])
    lst.start()
    assert wait_for(lst, "waiting")["wake_phrase"] == config.WAKE_PHRASE
    lst.stop()


# ---- short phrases: no "I'm here, sir" ------------------------------------------------

import talk  # noqa: E402


@pytest.mark.parametrize("said,sent", [
    ("Jarvis, stop.", "stop."), ("Hey Jarvis, what time is it?", "what time is it?"),
    ("Thanks, Jarvis.", "Thanks"), ("mute", "mute"), ("Jarvis", ""), ("Hey Jarvis.", ""),
])
def test_wake_word_is_stripped_before_the_model_sees_it(said, sent):
    assert talk.strip_wake(said) == sent


@pytest.mark.parametrize("said", ["Jarvis.", "hey jarvis", "Um.", "Hmm", "Jarvis, um", "you"])
def test_name_or_filler_alone_is_not_a_request(said):
    assert not talk.is_real_speech(said)


@pytest.mark.parametrize("said", ["Stop.", "Mute", "Yes", "No", "Okay", "Jarvis, louder"])
def test_short_requests_still_count(said):
    assert talk.is_real_speech(said)


def test_name_alone_opens_the_conversation_silently():
    lst, made = make_listener(["Jarvis.", "Jarvis, stop."])
    lst.start()
    heard = wait_for(lst, "heard")
    assert heard["text"] == "stop."  # the model gets the request, not his name
    assert len([e for e in lst.events_after(0, 0) if e["type"] == "heard"]) == 1
    assert lst.conversation
    lst.stop()


def test_a_new_page_does_not_replay_old_speech():
    """Refreshing the HUD used to replay every 'heard' event, so he re-answered the old chat."""
    lst, _ = make_listener(["Who made you?", "Goats."])
    lst.start(wake_word=False)
    wait_for(lst, "heard")
    time.sleep(0.3)
    chunks = "".join(lst.sse(0, max_seconds=0.3))
    assert '"heard"' not in chunks
    # a page that outlived a server restart (its last id is ahead of ours) also starts fresh
    chunks = "".join(lst.sse(10_000, max_seconds=0.3))
    assert '"heard"' not in chunks and '"state"' in chunks
    lst.stop()


def test_a_connected_page_still_gets_new_speech():
    lst, _ = make_listener([])
    lst.start(wake_word=False)
    wait_for(lst, "waiting")
    with lst._cond:
        last = lst._next_id - 1
    lst.emit("heard", text="what time is it")
    chunks = "".join(lst.sse(last, max_seconds=0.3))
    assert "what time is it" in chunks
    lst.stop()


# ---- echo guard: he must not answer his own voice -----------------------------------

import listener as listener_module  # noqa: E402


def test_his_own_sentence_coming_back_through_the_mic_is_ignored(monkeypatch):
    monkeypatch.setattr(listener_module, "_SPOKEN", listener_module.deque(maxlen=60))
    listener_module.note_spoken("The weather in Sydney is overcast with a temperature of 17 degrees, sir.")
    listener_module.note_spoken("Very good, sir.")
    assert listener_module.is_echo("the weather in Sydney is overcast with a temperature of 17 degrees")
    assert listener_module.is_echo("Very good, sir.")
    assert not listener_module.is_echo("What's the weather tomorrow?")
    assert not listener_module.is_echo("Good")
    assert not listener_module.is_echo("Set a timer for 17 minutes")


def test_echo_is_dropped_not_answered(monkeypatch):
    monkeypatch.setattr(listener_module, "_SPOKEN", listener_module.deque(maxlen=60))
    listener_module.note_spoken("It is eight o'clock in the evening, sir.")
    lst, _ = make_listener(["It is eight o'clock in the evening, sir.", "What time is it?"])
    lst.start(wake_word=False)
    heard = wait_for(lst, "heard")
    assert heard["text"] == "What time is it?"
    assert any(e["type"] == "echo" for e in lst.events_after(0, 0))
    lst.stop()


# ---- a mute must never outlive the page that set it ---------------------------------

def test_a_new_page_lifts_a_mute_left_by_a_closed_page():
    lst, made = make_listener(["What time is it?"])
    lst.start(wake_word=False)
    wait_for(lst, "waiting")
    lst.mute(True)                    # the old page muted him while he answered, then went away
    assert made["rec"].mic_on is False
    next(lst.sse(last_id=0))          # a new page connects
    assert made["rec"].mic_on is True and lst._muted is False
    lst.stop()


def test_a_connected_page_reconnecting_keeps_its_mute():
    lst, made = make_listener([])
    lst.start(wake_word=False)
    wait_for(lst, "waiting")
    lst.mute(True)
    newest = lst._next_id - 1
    next(lst.sse(last_id=newest), None)   # the same page's automatic reconnect
    assert made["rec"].mic_on is False
    lst.stop()


def test_a_forgotten_mute_times_out(monkeypatch):
    monkeypatch.setattr(listener_module, "MUTE_MAX_SECONDS", 0.5)
    lst, made = make_listener([])
    lst.start(wake_word=False)
    wait_for(lst, "waiting")
    lst.mute(True)
    wait_for(lst, "unmuted", timeout=4)
    assert made["rec"].mic_on is True
    lst.stop()


def test_follow_ups_that_reuse_his_words_are_not_echoes(monkeypatch):
    monkeypatch.setattr(listener_module, "_SPOKEN", listener_module.deque(maxlen=60))
    for said in ("It is Sunday, 15 November 2026, at 12:43 PM, sir.", "Teal, sir.",
                 "The volume has been reduced, sir.",
                 "A long day, sir. I hope your tea is strong enough to match your resolve.",
                 "I'm afraid the weather service is unavailable, sir. Perhaps we can talk about something else.",
                 "Ephemeral means lasting for a short period of time, sir."):
        listener_module.note_spoken(said)
    assert not listener_module.is_echo("how long is a short period of time")
    assert not listener_module.is_echo("what time is it now sir")
    # the real thing still counts, including the end of one sentence and the start of the next
    assert listener_module.is_echo("lasting for a short period of time sir")
    assert listener_module.is_echo("teal sir the volume has been reduced")


# ---- know sir's voice -----------------------------------------------------------------

def test_another_voice_is_ignored_when_enrolled(monkeypatch):
    import voiceid
    monkeypatch.setattr(voiceid, "active", lambda: True)
    verdicts = iter([False, True])
    monkeypatch.setattr(voiceid, "is_sir", lambda audio: next(verdicts))
    lst, made = make_listener(["Turn the volume up", "What time is it?"])
    lst.start(wake_word=False)
    heard = wait_for(lst, "heard")
    assert heard["text"] == "What time is it?"
    assert any(e["type"] == "not_sir" and e["text"] == "Turn the volume up" for e in lst.events_after(0, 0))
    lst.stop()


def test_cant_tell_or_not_enrolled_never_locks_sir_out(monkeypatch):
    import voiceid
    monkeypatch.setattr(voiceid, "active", lambda: True)
    monkeypatch.setattr(voiceid, "is_sir", lambda audio: None)   # too short to judge
    lst, _ = make_listener(["yes"])
    lst.start(wake_word=False)
    assert wait_for(lst, "heard")["text"] == "yes"
    lst.stop()
    monkeypatch.setattr(voiceid, "active", lambda: False)        # not enrolled
    assert listener_module.not_sirs_voice(type("R", (), {"last_transcription_bytes": [0.1]})()) is False
