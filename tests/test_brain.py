"""Offline tests for the thinking loop, using a scripted fake model instead of Ollama."""
from types import SimpleNamespace

import pytest

import brain
import config
import memory


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "workspace")
    memory.init_db()


def say(text):
    return SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))


def call(name, **arguments):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[SimpleNamespace(function=fn)]))


def script(monkeypatch, *responses):
    """Make brain.chat return the given responses in order, and record what it was sent."""
    queue, seen = list(responses), []

    def fake_chat(**kwargs):
        seen.append([m if isinstance(m, dict) else {"role": "assistant"} for m in kwargs["messages"]])
        response = queue.pop(0)
        return iter([response]) if kwargs.get("stream") else response

    monkeypatch.setattr(brain, "chat", fake_chat)
    return seen


def test_plain_reply_is_saved_to_history(monkeypatch):
    script(monkeypatch, say("Good evening, sir."))
    assert brain.respond("hello", "s") == "Good evening, sir."
    assert [m["role"] for m in memory.recent_messages("s")] == ["user", "assistant"]


def test_tool_result_is_fed_back_to_the_model(monkeypatch):
    seen = script(monkeypatch, call("search_notes", query="zebra"), say("Nothing found, sir."))
    assert brain.respond("find my zebra note", "s") == "Nothing found, sir."
    assert seen[1][-1] == {"role": "tool", "tool_name": "search_notes", "content": "No notes matched."}


def test_false_memory_claim_is_caught_and_fixed(monkeypatch):
    seen = script(
        monkeypatch,
        say("I have remembered that, sir."),
        call("remember_fact", fact="Sir's lucky number is 47."),
        say("Noted, sir."),
    )
    assert brain.respond("my lucky number is 47, keep it in mind", "s") == "Noted, sir."
    assert memory.all_facts() == ["Sir's lucky number is 47."]
    assert seen[1][-1]["content"] == brain.NUDGE


def test_repeated_false_claim_is_replaced_with_an_honest_answer(monkeypatch):
    script(monkeypatch, say("I have remembered that."), say("I have remembered that."))
    assert brain.respond("remember x", "s") == brain.MEMORY_FAILED
    assert memory.all_facts() == []


def test_recalling_a_fact_does_not_trigger_the_guard(monkeypatch):
    seen = script(monkeypatch, say("I remembered it, sir: 47."))
    assert brain.respond("what is my lucky number?", "s") == "I remembered it, sir: 47."
    assert len(seen) == 1


def test_tool_loop_is_capped(monkeypatch):
    script(monkeypatch, *[call("get_current_time") for _ in range(config.MAX_TOOL_ROUNDS)])
    assert "gave up" in brain.respond("loop forever", "s")
    assert len(memory.recent_audit()) == config.MAX_TOOL_ROUNDS


def test_facts_reach_the_system_prompt_in_new_conversations(monkeypatch):
    memory.add_fact("Sir's lucky number is 47.")
    seen = script(monkeypatch, say("47, sir."))
    brain.respond("what is my lucky number?", "brand-new")
    assert "lucky number is 47" in seen[0][0]["content"]


@pytest.mark.parametrize("reply,expected", [
    ("It is noon, sir. Is there anything else I can assist you with?", "It is noon, sir."),
    ("Done. How else may I help you today?", "Done."),
    ("Shall I open notepad?", "Shall I open notepad?"),
    ("Would you like the weather for Melbourne as well?", "Would you like the weather for Melbourne as well?"),
])
def test_stock_offer_is_stripped_but_real_questions_stay(reply, expected):
    assert brain.strip_stock_offer(reply) == expected


def test_stock_offer_is_stripped_end_to_end(monkeypatch):
    script(monkeypatch, say("Good evening, sir. Is there anything else I can assist you with?"))
    assert brain.respond("hi", "s") == "Good evening, sir."


def test_system_prompt_states_who_built_him():
    prompt = brain._system_prompt()
    assert "built by sir" in prompt and "bughunter3333-lab/J.A.R.V.I.S" in prompt
    assert "not related to Microsoft" in prompt
