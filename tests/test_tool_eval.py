"""Offline tests for the tool-picking eval's scoring (the real run needs Ollama)."""
from types import SimpleNamespace

import tools
from evals import tool_eval


def fake_chat_from(answers):
    """answers: {message: tool name or None}; returns a fake ollama chat."""
    def chat(**kwargs):
        msg = kwargs["messages"][-1]["content"]
        name = answers.get(msg)
        calls = [SimpleNamespace(function=SimpleNamespace(name=name, arguments={}))] if name else None
        return SimpleNamespace(message=SimpleNamespace(content="", tool_calls=calls))
    return chat


def test_every_expected_tool_exists():
    for _, expected, _ in tool_eval.CASES:
        for name in (expected if isinstance(expected, tuple) else (expected,)):
            assert name is None or name in tools.TOOLS


def test_grade():
    assert tool_eval.grade("calculate", None, [{"name": "calculate", "args": {}}])[0]
    assert not tool_eval.grade("calculate", None, [])[0]
    assert tool_eval.grade(None, None, [])[0]
    assert not tool_eval.grade(None, None, [{"name": "web_search", "args": {}}])[0]
    assert tool_eval.grade("pc_control", {"action": "mute"}, [{"name": "pc_control", "args": {"action": "mute"}}])[0]
    assert not tool_eval.grade("pc_control", {"action": "mute"}, [{"name": "pc_control", "args": {"action": "volume_up"}}])[0]
    assert tool_eval.grade("web_search", {"news": True}, [{"name": "web_search", "args": {"news": "true"}}])[0]
    assert tool_eval.grade(("look_up", "web_search"), None, [{"name": "web_search", "args": {}}])[0]
    assert not tool_eval.grade(("look_up", "web_search"), None, [])[0]


def test_run_scores_and_reports_mistakes():
    answers = {m: e for m, e, _ in tool_eval.CASES if not _ and not isinstance(e, tuple)}
    answers["Who was Alan Turing?"] = "web_search"  # one deliberate mistake
    s = tool_eval.run(only="look_up", chat=fake_chat_from(answers), show=lambda *_: None)
    assert s["total"] == 2
    assert ("look_up -> web_search", 1) in s["common_mistakes"]
