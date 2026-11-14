"""JARVIS-X web API. Start with:  uvicorn main:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

import brain
import memory

app = FastAPI(title="JARVIS-X")
memory.init_db()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session: str = Field(default="default", max_length=64)


@app.get("/")
def status():
    return {"status": "online", "message": "Good evening. JARVIS-X is running."}


@app.post("/chat")
def talk(request: ChatRequest):
    return {"reply": brain.respond(request.message, request.session)}


@app.get("/audit")
def audit(limit: int = 20):
    """Recent tool calls, newest first. Shows what Jarvis did and what was refused."""
    return memory.recent_audit(min(max(limit, 1), 200))
