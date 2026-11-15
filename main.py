"""JARVIS-X web API. Start with:  uvicorn main:app --reload"""
import io
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import brain
import config
import doctor
import approvals
import memory
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    memory.init_db()
    reminders.init_db()
    reminders.start_checker()
    if os.environ.get("JARVIS_NO_VOICE_WARMUP") != "1":
        threading.Thread(target=_warm_up_voice, daemon=True).start()
    yield


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


@app.get("/health")
def health():
    """Quick doctor checks (Ollama, model, model folder, data folder, voice model)."""
    return doctor.summary(doctor.run_checks(quick=True))


@app.get("/hud", include_in_schema=False)
def hud():
    """The holographic interface. Served from here so the page and the API share one origin."""
    return FileResponse(HUD_PAGE, media_type="text/html")


@app.post("/chat")
def talk(request: ChatRequest):
    try:
        reply = brain.respond(request.message, request.session)
        return {"reply": reply, "approval": approvals.pending(request.session)}
    except Exception as exc:
        # 503 = the brain is unavailable; the detail tells the HUD exactly why.
        raise HTTPException(status_code=503, detail=explain(exc)) from exc


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
    reminders.fire_due()
    return reminders.take_unannounced()


@app.get("/stats")
def stats(limit: int = 100):
    """Latency and tool usage over the most recent requests (from the traces table)."""
    return memory.trace_stats(min(max(limit, 1), 1000))
