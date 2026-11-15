"""Offline tests for human approval of risky actions."""
from datetime import datetime, timedelta

import pytest

import approvals
import brain
import config
import memory
import pc
import tools
from tests.test_brain import call, say, script


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "workspace")
    memory.init_db()


@pytest.fixture
def opened(monkeypatch):
    urls = []
    monkeypatch.setattr(pc.webbrowser, "open", lambda u, new=0: urls.append(u))
    return urls


def test_risky_action_is_parked_not_run(opened):
    out = tools.run_tool("pc_control", {"action": "open_website", "url": "youtube.com"}, "s")
    assert out.startswith("WAITING FOR APPROVAL") and opened == []
    assert approvals.pending("s")["summary"] == "open youtube.com in the browser"
    assert memory.recent_audit()[0]["ok"] is False


def test_safe_actions_dont_ask():
    assert approvals.needs_approval("pc_control", {"action": "volume_up"}) is False
    assert approvals.needs_approval("calculate", {"expression": "1+1"}) is False
    assert approvals.needs_approval("clipboard", {"action": "read"}) is False
    assert approvals.needs_approval("create_note", {"title": "x"}) is True


@pytest.mark.parametrize("said,verdict", [
    ("yes", "approve"), ("Yes please", "approve"), ("go ahead", "approve"), ("ok", "approve"),
    ("do it, Jarvis", "approve"), ("no", "deny"), ("cancel that", "deny"), ("don't", "deny"),
    ("what's the weather", None), ("yes but first tell me the time and then open it", None),
])
def test_decision(said, verdict):
    assert approvals.decision(said) == verdict


def test_full_flow_yes_runs_it_in_code(monkeypatch, opened):
    script(monkeypatch, call("pc_control", action="open_website", url="youtube.com"),
           say("Shall I open YouTube, sir?"))
    assert brain.respond("open youtube", "s") == "Shall I open YouTube, sir?"
    assert opened == []
    # The model is not consulted for the yes: brain.chat would fail if it were called.
    monkeypatch.setattr(brain, "chat", lambda **k: (_ for _ in ()).throw(AssertionError("model called")))
    assert brain.respond("yes", "s") == "Done, sir. Opened youtube.com in the browser."
    assert opened == ["https://youtube.com"] and approvals.pending("s") is None


def test_full_flow_no_cancels(monkeypatch, opened):
    script(monkeypatch, call("pc_control", action="lock_screen"), say("Shall I lock the screen, sir?"))
    brain.respond("lock it", "s")
    assert brain.respond("no", "s") == "Very good, sir. I've left it."
    assert approvals.pending("s") is None
    assert any(e["result"] == "Denied by sir." for e in memory.recent_audit())


def test_changing_the_subject_drops_the_request(monkeypatch, opened):
    script(monkeypatch, call("pc_control", action="open_website", url="evil.example"), say("Shall I?"),
           say("It is noon, sir."), say("Yes to what, sir?"))
    brain.respond("open evil.example", "s")
    assert brain.respond("what time is it", "s") == "It is noon, sir."
    assert brain.respond("yes", "s") == "Yes to what, sir?"
    assert opened == []


def test_request_expires(opened):
    tools.run_tool("pc_control", {"action": "lock_screen"}, "s")
    later = datetime.now() + timedelta(seconds=config.APPROVAL_TIMEOUT + 1)
    assert approvals.pending("s", now=later) is None


def test_approval_is_per_conversation(opened):
    tools.run_tool("pc_control", {"action": "open_website", "url": "a.example"}, "hud")
    assert approvals.pending("other") is None


def test_model_cannot_approve_itself(monkeypatch, opened):
    """Even if a web page tells the model to pass approved=True, it's just an unknown argument."""
    out = tools.run_tool("pc_control", {"action": "open_website", "url": "x.example", "approved": True}, "s")
    assert out.startswith("WAITING FOR APPROVAL") and opened == []


def test_restart_expires_waiting_approvals(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "a.db")
    memory.init_db()
    approvals.request("hud", "create_note", {"title": "copyright.txt"})
    assert approvals.pending("hud")
    approvals.expire_all()
    assert approvals.pending("hud") is None
