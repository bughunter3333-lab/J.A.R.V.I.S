"""Offline tests for memory and tools. No model or network needed.  Run:  pytest -q"""
import pytest

import config
import memory
import tools


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point every test at a throwaway database and workspace."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "workspace")
    memory.init_db()


def test_history_is_per_session_and_ordered():
    memory.add_message("a", "user", "one")
    memory.add_message("a", "assistant", "two")
    memory.add_message("b", "user", "other")
    assert [m["content"] for m in memory.recent_messages("a")] == ["one", "two"]


def test_history_limit_keeps_newest():
    for i in range(5):
        memory.add_message("a", "user", str(i))
    assert [m["content"] for m in memory.recent_messages("a", limit=2)] == ["3", "4"]


def test_facts_are_deduplicated():
    memory.add_fact("Sir studies cybersecurity.")
    memory.add_fact("Sir studies cybersecurity.")
    assert memory.all_facts() == ["Sir studies cybersecurity."]


def test_open_app_allowed(monkeypatch):
    launched = []
    monkeypatch.setattr(tools.subprocess, "Popen", lambda cmd: launched.append(cmd))
    assert tools.run_tool("open_app", {"name": "Notepad"}) == "Opened notepad."
    assert launched == [["notepad.exe"]]


@pytest.mark.parametrize("name", ["cmd", "powershell", "notepad.exe & calc", "C:\\Windows\\System32\\cmd.exe"])
def test_open_app_refuses_anything_off_the_list(monkeypatch, name):
    launched = []
    monkeypatch.setattr(tools.subprocess, "Popen", lambda cmd: launched.append(cmd))
    assert tools.run_tool("open_app", {"name": name}).startswith("Refused")
    assert launched == []


def test_create_note_and_search():
    assert tools.run_tool("create_note", {"title": "Shopping", "content": "milk and eggs"}, approved=True) == "Saved note 'Shopping.txt'."
    assert (config.WORKSPACE / "Shopping.txt").read_text(encoding="utf-8") == "milk and eggs"
    assert "milk" in tools.run_tool("search_notes", {"query": "EGGS"})
    assert tools.run_tool("search_notes", {"query": "unicorn"}) == "No notes matched."


@pytest.mark.parametrize("title", ["..\\..\\evil", "../../evil", "C:\\Windows\\evil", "sub/dir/evil"])
def test_create_note_cannot_escape_workspace(title):
    tools.run_tool("create_note", {"title": title, "content": "x"}, approved=True)
    written = [p for p in config.DATA_DIR.rglob("*.txt")]
    assert all(config.WORKSPACE in p.parents for p in written)
    assert not (config.DATA_DIR.parent / "evil.txt").exists()


def test_create_note_never_overwrites():
    tools.run_tool("create_note", {"title": "keep", "content": "original"}, approved=True)
    assert tools.run_tool("create_note", {"title": "keep", "content": "replaced"}, approved=True).startswith("Refused")
    assert (config.WORKSPACE / "keep.txt").read_text(encoding="utf-8") == "original"


def test_unknown_tool_and_bad_arguments_do_not_crash():
    assert tools.run_tool("delete_everything", {}).startswith("Refused")
    assert tools.run_tool("open_app", {"wrong": "arg"}).startswith("Failed")


def test_every_call_is_audited_including_refusals():
    tools.run_tool("get_current_time", {})
    tools.run_tool("open_app", {"name": "cmd"})
    log = memory.recent_audit()
    assert [(e["tool"], e["ok"]) for e in log] == [("open_app", False), ("get_current_time", True)]
