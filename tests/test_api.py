"""Offline tests for the web API, with the model replaced by a stub."""
import pytest
from fastapi.testclient import TestClient

import brain
import config
import main
import memory


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "workspace")
    memory.init_db()
    monkeypatch.setattr(brain, "respond", lambda message, session="default": f"echo:{session}:{message}")
    return TestClient(main.app)


def test_status(client):
    assert client.get("/").json()["status"] == "online"


def test_chat_passes_message_and_session(client):
    reply = client.post("/chat", json={"message": "hi", "session": "abc"}).json()
    assert reply == {"reply": "echo:abc:hi"}


def test_chat_rejects_empty_and_oversized_messages(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 4001}).status_code == 422


def test_audit_endpoint_lists_tool_calls(client):
    memory.log_tool("open_app", {"name": "cmd"}, "Refused: not on the allow-list.", False)
    entry = client.get("/audit").json()[0]
    assert entry["tool"] == "open_app" and entry["ok"] is False


def test_hud_page_is_served(client):
    page = client.get("/hud")
    assert page.status_code == 200 and "J.A.R.V.I.S." in page.text


def test_speak_returns_wav(client, monkeypatch):
    import numpy as np

    import voice

    monkeypatch.setattr(voice, "synthesize", lambda text: np.zeros(2400, dtype=np.float32))
    audio = client.post("/speak", json={"text": "Good evening, sir."})
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.content[:4] == b"RIFF"


def test_speak_rejects_empty_and_oversized_text(client):
    assert client.post("/speak", json={"text": ""}).status_code == 422
    assert client.post("/speak", json={"text": "x" * 2001}).status_code == 422


def test_chat_failure_is_503_with_a_readable_reason(client, monkeypatch):
    def broken(message, session="default"):
        raise Exception("model 'qwen3:8b' not found (status code: 404)")
    monkeypatch.setattr(brain, "respond", broken)
    res = client.post("/chat", json={"message": "hi"})
    assert res.status_code == 503 and "doctor" in res.json()["detail"]


def test_health_endpoint(client, monkeypatch):
    import doctor
    monkeypatch.setattr(doctor, "_ollama_models", lambda: None)
    body = client.get("/health").json()
    assert body["status"] == "fail"
    assert any(c["name"] == "Ollama" and c["status"] == "fail" for c in body["checks"])


def test_stats_endpoint(client):
    memory.add_trace("s", "m", 120, 1, ["get_current_time"], True)
    memory.add_trace("s", "m", 300, 2, [], False, "boom")
    body = client.get("/stats").json()
    assert body["requests"] == 2 and body["errors"] == 1
    assert body["tools_used"] == {"get_current_time": 1}


def test_forget_history_clears_only_that_session(client):
    memory.add_message("hud", "user", "who made you")
    memory.add_message("hud", "assistant", "Microsoft, sir.")
    memory.add_message("other", "user", "keep me")
    memory.add_fact("Sir studies cybersecurity.")
    assert client.delete("/history/hud").json() == {"session": "hud", "deleted": 2}
    assert memory.recent_messages("hud") == []
    assert len(memory.recent_messages("other")) == 1 and memory.all_facts()
