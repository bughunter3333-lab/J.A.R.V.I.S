"""Local voice for the HUD: the server listens to the PC's microphone itself.

The HUD switches it on with POST /listen/start. The same RealtimeSTT + "hey jarvis"
wake word as talk.py runs in a background thread inside the server, so nothing is
sent to Google and it works in any browser (Firefox included). What it hears is
pushed to the HUD over GET /listen/events (server-sent events); the HUD then sends
it through /chat/stream and speaks the reply. While Jarvis is speaking the HUD
mutes the microphone (POST /listen/mute) so he doesn't hear and answer himself.
"""
import json
import threading
import time
from collections import deque

import talk


class Listener:
    def __init__(self, make_recorder=talk.make_recorder):
        self._make_recorder = make_recorder
        self._recorder = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._events = deque(maxlen=200)
        self._next_id = 1
        self.state = "off"  # off, loading, waiting, listening, error
        self.wake_word = True

    # ---------------------------------------------------------------- events

    def emit(self, kind: str, **data) -> None:
        with self._cond:
            event = {"id": self._next_id, "type": kind, **data}
            self._next_id += 1
            self._events.append(event)
            if kind in ("loading", "waiting", "listening", "off", "error"):
                self.state = kind
            self._cond.notify_all()

    def events_after(self, last_id: int, timeout: float = 15.0) -> list[dict]:
        """Events newer than last_id; waits up to `timeout` seconds for one to arrive."""
        with self._cond:
            if not any(e["id"] > last_id for e in self._events):
                self._cond.wait(timeout)
            return [e for e in self._events if e["id"] > last_id]

    # ---------------------------------------------------------------- control

    def start(self, wake_word: bool = True) -> str:
        with self._lock:
            if self._running:
                return self.state
            self._running, self.wake_word = True, wake_word
            self.emit("loading")
            self._thread = threading.Thread(target=self._run, daemon=True, name="listener")
            self._thread.start()
            return "loading"

    def stop(self) -> None:
        with self._lock:
            self._running = False
            recorder, self._recorder = self._recorder, None
        if recorder:
            try:
                recorder.abort()
                recorder.shutdown()
            except Exception:
                pass
        self.emit("off")

    def mute(self, muted: bool) -> None:
        if self._recorder:
            try:
                self._recorder.set_microphone(not muted)
            except Exception:
                pass

    # ---------------------------------------------------------------- the loop

    def _run(self) -> None:
        try:
            recorder = self._make_recorder(
                use_wake_word=self.wake_word,
                on_wakeword_detected=lambda: self.emit("wake"),
                on_recording_start=lambda: self.emit("listening"),
                on_recording_stop=lambda: self.emit("transcribing"),
            )
        except Exception as exc:
            self._running = False
            self.emit("error", detail=f"Couldn't start the microphone: {type(exc).__name__}: {exc}")
            return
        self._recorder = recorder
        self.emit("waiting", wake_word=self.wake_word)
        while self._running:
            try:
                heard = recorder.text()
            except Exception as exc:
                if self._running:
                    self.emit("error", detail=f"Listening failed: {type(exc).__name__}: {exc}")
                    self._running = False
                break
            if not self._running:
                break
            if heard and talk.is_real_speech(heard):
                self.emit("heard", text=heard.strip())
            self.emit("waiting", wake_word=self.wake_word)
            time.sleep(0.05)

    def sse(self, last_id: int = 0, max_seconds: float = 25.0):
        """Server-sent events for the HUD. Each stream ends after max_seconds and the browser
        reconnects by itself (sending Last-Event-ID), so a reload of uvicorn never hangs
        waiting for an open stream."""
        if last_id == 0:
            yield f"data: {json.dumps({'id': 0, 'type': 'state', 'state': self.state})}\n\n"
        yield "retry: 500\n\n"
        ends = time.monotonic() + max_seconds
        while time.monotonic() < ends:
            events = self.events_after(last_id, timeout=min(5.0, max(0.1, ends - time.monotonic())))
            if not events:
                yield ": keep-alive\n\n"
                continue
            for event in events:
                last_id = event["id"]
                yield f"id: {event['id']}\ndata: {json.dumps(event)}\n\n"

listener = Listener()
