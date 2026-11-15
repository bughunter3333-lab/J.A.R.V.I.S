"""Local voice for the HUD: the server listens to the PC's microphone itself.

The HUD switches it on with POST /listen/start. The same RealtimeSTT + wake word ("Jarvis")
wake word as talk.py runs in a background thread inside the server, so nothing is
sent to Google and it works in any browser (Firefox included). What it hears is
pushed to the HUD over GET /listen/events (server-sent events); the HUD then sends
it through /chat/stream and speaks the reply. While Jarvis is speaking the HUD
mutes the microphone (POST /listen/mute) so he doesn't hear and answer himself.

Conversation mode: the wake word starts a conversation and the wake word is switched
off, so every follow-up goes straight to him. "That's all, thank you" (see
config.CONVERSATION_END_PHRASES) or CONVERSATION_IDLE_TIMEOUT seconds of silence ends
it and the wake word is needed again.
"""
import json
import re
import threading
import time
from collections import deque

import config
import talk


def is_conversation_end(heard: str) -> bool:
    """'That's all, thank you', 'thanks Jarvis, that will be all', 'goodbye'... (short phrases only,
    so 'that's all I wanted to ask about the weather' doesn't end anything)."""
    text = re.sub(r"[^a-z' ]", " ", heard.lower().replace("\u2019", "'"))
    text = re.sub(r"\b(thank you|thanks|thank|cheers|sir|jarvis|ok(ay)?|alright|please|for now|very much|so much)\b", " ", text)
    text = " ".join(text.split())
    return bool(text) and len(text.split()) <= 5 and any(
        text == p or text.startswith(p + " ") or text.endswith(" " + p) for p in config.CONVERSATION_END_PHRASES)


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
        self.conversation = False  # True between the wake word and "that's all"
        self._muted = False
        self._last_activity = time.monotonic()
        self._watchdog = None

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
            self.conversation = False
            recorder, self._recorder = self._recorder, None
        if recorder:
            try:
                recorder.abort()
                recorder.shutdown()
            except Exception:
                pass
        self.emit("off")

    def mute(self, muted: bool) -> None:
        self._muted = muted
        if not muted:
            self._last_activity = time.monotonic()  # the idle clock starts when he stops talking
        if self._recorder:
            try:
                self._recorder.set_microphone(not muted)
            except Exception:
                pass

    # ---------------------------------------------------------------- conversation mode

    def _set_wake_word(self, needed: bool) -> None:
        """Switch RealtimeSTT between "wait for the wake word" and "take whatever is said".
        The recorder checks this flag on every audio chunk, so it applies immediately."""
        if self._recorder is not None and hasattr(self._recorder, "use_wake_words"):
            self._recorder.use_wake_words = needed

    def _begin_conversation(self) -> None:
        if self.conversation or not self.wake_word:
            return
        self.conversation = True
        self._last_activity = time.monotonic()
        self._set_wake_word(False)
        self.emit("conversation", active=True)

    def end_conversation(self, reason: str) -> None:
        if not self.conversation:
            return
        self.conversation = False
        self._set_wake_word(True)
        farewell = config.CONVERSATION_FAREWELL if reason == "phrase" else ""
        self.emit("conversation", active=False, reason=reason, farewell=farewell)

    def _watch_idle(self) -> None:
        """End a conversation nobody is talking in (not while he is speaking)."""
        while self._running:
            time.sleep(1.0)
            limit = config.CONVERSATION_IDLE_TIMEOUT
            if (self.conversation and limit and not self._muted
                    and time.monotonic() - self._last_activity > limit):
                self.end_conversation("idle")

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
        self._watchdog = threading.Thread(target=self._watch_idle, daemon=True, name="listener-idle")
        self._watchdog.start()
        self.emit("waiting", wake_word=self.wake_word, wake_phrase=config.WAKE_PHRASE)
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
                self._last_activity = time.monotonic()
                if self.conversation and is_conversation_end(heard):
                    self.end_conversation("phrase")
                else:
                    self._begin_conversation()  # wake word + request starts one
                    self.emit("heard", text=talk.strip_wake(heard))
            elif heard and self.wake_word and not self.conversation:
                # Only "Jarvis" (or "um"): sir is about to ask something, so open the
                # conversation silently and take the next thing he says without the wake word.
                self._last_activity = time.monotonic()
                self._begin_conversation()
            self.emit("waiting", wake_word=self.wake_word and not self.conversation,
                      wake_phrase=config.WAKE_PHRASE)
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
