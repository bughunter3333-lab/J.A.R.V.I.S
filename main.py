"""JARVIS-X web API. Start with:  uvicorn main:app --reload"""
import io

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import brain
import config
import memory

app = FastAPI(title="JARVIS-X")
memory.init_db()

HUD_PAGE = config.BASE_DIR / "ui" / "index.html"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session: str = Field(default="default", max_length=64)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/")
def status():
    return {"status": "online", "message": "Good evening. JARVIS-X is running."}


@app.get("/hud", include_in_schema=False)
def hud():
    """The holographic interface. Served from here so the page and the API share one origin."""
    return FileResponse(HUD_PAGE, media_type="text/html")


@app.post("/chat")
def talk(request: ChatRequest):
    return {"reply": brain.respond(request.message, request.session)}


@app.post("/speak")
def speak(request: SpeakRequest):
    """Jarvis's voice as a WAV file, for the HUD to play (and animate the core with)."""
    import soundfile as sf

    import voice

    buffer = io.BytesIO()
    sf.write(buffer, voice.synthesize(request.text), voice.SAMPLE_RATE, format="WAV")
    return Response(buffer.getvalue(), media_type="audio/wav")


@app.get("/audit")
def audit(limit: int = 20):
    """Recent tool calls, newest first. Shows what Jarvis did and what was refused."""
    return memory.recent_audit(min(max(limit, 1), 200))
