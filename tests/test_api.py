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
