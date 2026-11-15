"""Offline tests for local listening, with a fake RealtimeSTT recorder (no microphone)."""
import json
import queue
import time

from fastapi.testclient import TestClient

from listener import Listener


class FakeRecorder:
    def __init__(self, phrases, **callbacks):
        self.phrases, self.cb = queue.Queue(), callbacks
        for p in phrases:
            self.phrases.put(p)
        self.mic_on, self.closed = True, False

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
    assert json.loads(chunks[0][len("data: "):])["type"] == "state"


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
