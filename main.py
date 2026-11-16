"""JARVIS-X web API. Start with:  uvicorn main:app --reload"""
import io
import json
import os
import sys
import time
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import brain
import config
import doctor
import approvals
import memory
import security
import reminders

HUD_PAGE = config.BASE_DIR / "ui" / "index.html"
_voice_lock = threading.Lock()  # Kokoro does one sentence at a time


def _warm_up_voice() -> None:
    """Load Kokoro in the background so the HUD's first reply isn't silent."""
    try:
        import voice

        with _voice_lock:
            voice.synthesize("Ready.")
        print("Voice loaded.")
    except Exception as exc:  # the API still works without a voice
        print(f"Voice not available yet: {type(exc).__name__}: {exc}")


def _warm_up_model() -> None:
    """Load the language model in the background so the first reply isn't a cold start."""
    try:
        brain.warm_up()
        print("Model loaded.")
    except Exception as exc:  # Ollama not running yet: the first request will load it
        print(f"Model not loaded yet: {type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    memory.init_db()
    approvals.expire_all()  # an approval box from before a restart must not come back
    reminders.init_db()
    reminders.start_checker()
    if os.environ.get("JARVIS_NO_VOICE_WARMUP") != "1":
        threading.Thread(target=_warm_up_voice, daemon=True).start()
    if os.environ.get("JARVIS_NO_MODEL_WARMUP") != "1" and "pytest" not in sys.modules:
        threading.Thread(target=_warm_up_model, daemon=True).start()
    yield
    from listener import listener
    listener.stop()  # release the microphone when the server stops or reloads


app = FastAPI(title="JARVIS-X", lifespan=lifespan)
memory.init_db()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session: str = Field(default="default", max_length=64)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def explain(exc: Exception) -> str:
    """Turn the usual brain failures into a sentence that says what to do."""
    text = str(exc)
    if "not found" in text and "model" in text:
        return f"Model {config.MODEL} isn't installed in Ollama. Run: python doctor.py"
    if isinstance(exc, ConnectionError) or "connect" in text.lower():
        return "Ollama isn't running. Open the Ollama app, then try again."
    return f"{type(exc).__name__}: {text[:200]}"


@app.get("/")
def status():
    return {"status": "online", "message": "Good evening. JARVIS-X is running."}


# Each server start is a fresh conversation for the HUD. Short-term chat history stays in
# the database but isn't sent back to the model; long-term facts (remember_fact) carry over.
HUD_SESSION = "hud-" + time.strftime("%Y%m%d-%H%M%S")


@app.get("/health")
def health():
    """Quick doctor checks (Ollama, model, model folder, data folder, voice model)."""
    return {**doctor.summary(doctor.run_checks(quick=True)), "session": HUD_SESSION}


@app.get("/hud", include_in_schema=False)
def hud():
    """The holographic interface. Served from here so the page and the API share one origin."""
    return FileResponse(HUD_PAGE, media_type="text/html")


class FeedbackRequest(BaseModel):
    turn: int
    rating: int = Field(..., description="1 = good reply, -1 = bad reply")
    correction: str = Field("", max_length=1000, description="what he should have said (optional)")


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Sir rates one reply (the 'turn' id from the reply's done event). Used to fine-tune him:
    python training\\build_dataset.py picks rated replies up."""
    if request.rating not in (1, -1):
        raise HTTPException(status_code=422, detail="rating must be 1 or -1")
    if not memory.rate_turn(request.turn, request.rating, security.redact(request.correction)):
        raise HTTPException(status_code=404, detail="No such reply.")
    return {"ok": True}


@app.get("/feedback")
def list_feedback():
    """Every rated reply, oldest first."""
    return memory.rated_turns()


@app.post("/chat")
def talk(request: ChatRequest):
    try:
        reply = brain.respond(request.message, request.session)
        return {"reply": reply, "approval": approvals.pending(request.session)}
    except Exception as exc:
        # 503 = the brain is unavailable; the detail tells the HUD exactly why.
        raise HTTPException(status_code=503, detail=explain(exc)) from exc


@app.post("/chat/stream")
def talk_stream(request: ChatRequest):
    """Same as /chat, streamed as one JSON object per line: status (which tool he's using),
    sentence (speak this now), then done (with any action waiting for approval) or error."""

    def events():
        try:
            from listener import note_spoken
            for event in brain.respond_events(request.message, request.session):
                if event["type"] == "sentence":
                    note_spoken(event["text"])  # so the mic can tell his voice from sir's
                if event["type"] == "done":
                    event["approval"] = approvals.pending(request.session)
                yield json.dumps(event) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "detail": explain(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/speak")
def speak(request: SpeakRequest):
    """Jarvis's voice as a WAV file, for the HUD to play (and animate the core with)."""
    import soundfile as sf

    import security
    import voice

    try:
        with _voice_lock:
            audio = voice.synthesize(security.redact(request.text))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Voice unavailable: {type(exc).__name__}: {exc}") from exc
    buffer = io.BytesIO()
    sf.write(buffer, audio, voice.SAMPLE_RATE, format="WAV")
    return Response(buffer.getvalue(), media_type="audio/wav")


@app.delete("/history/{session}")
def forget_history(session: str):
    """Clear one conversation so Jarvis stops repeating something wrong he said earlier.
    Remembered facts and the audit log are not touched."""
    return {"session": session, "deleted": memory.clear_session(session[:64])}


class ListenRequest(BaseModel):
    wake_word: bool = True


class MuteRequest(BaseModel):
    muted: bool


@app.post("/listen/start")
def listen_start(request: ListenRequest):
    """Start local listening on this PC's microphone (RealtimeSTT + the wake word)."""
    from listener import listener
    return {"state": listener.start(request.wake_word)}


@app.post("/listen/stop")
def listen_stop():
    from listener import listener
    listener.stop()
    return {"state": "off"}


@app.post("/listen/mute")
def listen_mute(request: MuteRequest):
    """The HUD mutes the mic while Jarvis speaks, so he doesn't answer himself."""
    from listener import listener
    listener.mute(request.muted)
    return {"muted": request.muted}


@app.get("/listen/events")
def listen_events(last_event_id: str | None = Header(default=None)):
    """Server-sent events: loading, waiting, wake, listening, transcribing, heard {text}, error, off."""
    from listener import listener
    last = int(last_event_id) if (last_event_id or "").isdigit() else 0
    return StreamingResponse(listener.sse(last), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/audit")
def audit(limit: int = 20):
    """Recent tool calls, newest first. Shows what Jarvis did and what was refused."""
    return memory.recent_audit(min(max(limit, 1), 200))


@app.get("/reminders")
def list_reminders():
    """Reminders still waiting to go off."""
    return reminders.pending()


@app.get("/reminders/due")
def due_reminders():
    """Reminders that have gone off and haven't been announced yet. The HUD polls this
    and says each one out loud; each is handed out only once."""
    from listener import note_spoken
    reminders.fire_due()
    due = reminders.take_unannounced()
    for item in due:
        note_spoken(str(item.get("message", "")))
    return due


@app.get("/stats")
def stats(limit: int = 100):
    """Latency and tool usage over the most recent requests (from the traces table)."""
    return memory.trace_stats(min(max(limit, 1), 1000))
