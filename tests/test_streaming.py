"""Offline tests for streamed replies (brain.respond_events and POST /chat/stream)."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import brain
import config
import memory


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "workspace")
    memory.init_db()


def chunk(text="", calls=None):
    return SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=calls))


def tool_call(name, **args):
    return [SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))]


def stream(monkeypatch, *rounds):
    """Each round is a list of chunks the fake model streams."""
    queue = list(rounds)
    monkeypatch.setattr(brain, "chat", lambda **k: iter(queue.pop(0)))


def test_sentences_arrive_as_they_finish(monkeypatch):
    stream(monkeypatch, [chunk("Good eve"), chunk("ning, sir. The time"), chunk(" is noon."),
                         chunk(" Is there anything else I can assist you with?")])
    events = list(brain.respond_events("hi", "s"))
    assert [e["text"] for e in events if e["type"] == "sentence"] == ["Good evening, sir.", "The time is noon."]
    assert {k: v for k, v in events[-1].items() if k != "turn"} == {"type": "done", "reply": "Good evening, sir. The time is noon."}
    assert isinstance(events[-1]["turn"], int)
    assert memory.recent_messages("s")[-1]["content"] == "Good evening, sir. The time is noon."


def test_status_event_for_each_tool(monkeypatch):
    stream(monkeypatch, [chunk("", tool_call("calculate", expression="2+2"))], [chunk("It is 4, sir.")])
    events = list(brain.respond_events("2+2?", "s"))
    assert {"type": "status", "text": "Calculating"} in events
    assert events[-1]["reply"] == "It is 4, sir."


def test_secret_is_redacted_before_it_is_streamed(monkeypatch):
    token = "ghp_" + "b" * 36
    stream(monkeypatch, [chunk(f"Your token is {token[:10]}"), chunk(f"{token[10:]}, sir.")])
    events = list(brain.respond_events("what's my token", "s"))
    assert all(token not in json.dumps(e) for e in events)


def test_remember_requests_are_held_back_until_checked(monkeypatch):
    """A false 'I've remembered' must never be streamed before the code catches it."""
    stream(monkeypatch, [chunk("I have remembered that, sir.")],
           [chunk("", tool_call("remember_fact", fact="Exam on Friday."))], [chunk("Noted, sir.")])
    sentences = [e["text"] for e in brain.respond_events("remember my exam is on Friday", "s") if e["type"] == "sentence"]
    assert sentences == ["Noted, sir."] and memory.all_facts() == ["Exam on Friday."]


def test_chat_stream_endpoint(monkeypatch):
    import main
    stream(monkeypatch, [chunk("", tool_call("get_current_time"))], [chunk("It is noon, sir.")])
    res = TestClient(main.app).post("/chat/stream", json={"message": "time?", "session": "hud"})
    lines = [json.loads(line) for line in res.text.splitlines()]
    assert lines[0] == {"type": "status", "text": "Checking the time"}
    assert lines[-1]["type"] == "done" and lines[-1]["reply"] == "It is noon, sir." and lines[-1]["approval"] is None


def test_chat_stream_reports_errors_as_an_event(monkeypatch):
    import main

    def broken(**k):
        raise ConnectionError("Failed to connect to Ollama")
    monkeypatch.setattr(brain, "chat", broken)
    res = TestClient(main.app).post("/chat/stream", json={"message": "hi"})
    last = json.loads(res.text.splitlines()[-1])
    assert last["type"] == "error" and "Ollama isn't running" in last["detail"]


def test_chat_stream_includes_pending_approval(monkeypatch):
    import main
    import pc
    monkeypatch.setattr(pc.webbrowser, "open", lambda u, new=0: None)
    stream(monkeypatch, [chunk("", tool_call("pc_control", action="lock_screen"))], [chunk("Shall I lock it, sir?")])
    res = TestClient(main.app).post("/chat/stream", json={"message": "lock", "session": "hud"})
    done = json.loads(res.text.splitlines()[-1])
    assert done["approval"]["summary"] == "lock the screen"


def test_empty_model_reply_is_never_silent(monkeypatch, tmp_path):
    import brain, config, memory
    from types import SimpleNamespace as NS
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "e.db")
    memory.init_db()
    monkeypatch.setattr(brain, "chat", lambda **kw: iter([NS(message=NS(content="", tool_calls=None))]))
    events = list(brain.respond_events("Goats.", "s"))
    assert {k: v for k, v in events[-1].items() if k != "turn"} == {"type": "done", "reply": brain.EMPTY_REPLY}
    assert isinstance(events[-1]["turn"], int)
    assert {"type": "sentence", "text": brain.EMPTY_REPLY} in events


def test_presence_filler_is_never_streamed(monkeypatch):
    import brain
    from types import SimpleNamespace
    monkeypatch.setattr(brain, "chat", lambda **kw: iter(
        [SimpleNamespace(message=SimpleNamespace(content="I am here, sir.", tool_calls=None))]))
    events = list(brain.respond_events("Stop", "filler"))
    assert [e["text"] for e in events if e["type"] == "sentence"] == [brain.ACKNOWLEDGE]
    assert {k: v for k, v in events[-1].items() if k != "turn"} == {"type": "done", "reply": brain.ACKNOWLEDGE}
    assert isinstance(events[-1]["turn"], int)
