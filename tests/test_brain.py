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


def test_remember_request_without_any_tool_call_is_retried(monkeypatch):
    """Found by evals/tool_eval.py: 'Keep in mind...' got a plain 'Noted, sir.' and nothing saved."""
    seen = script(
        monkeypatch,
        say("Noted, sir."),
        call("remember_fact", fact="Sir prefers tea over coffee."),
        say("Very good, sir."),
    )
    assert brain.respond("Keep in mind that I prefer tea over coffee", "s") == "Very good, sir."
    assert memory.all_facts() == ["Sir prefers tea over coffee."]
    assert seen[1][-1]["content"] == brain.NUDGE


def test_remember_request_that_still_isnt_saved_says_so(monkeypatch):
    script(monkeypatch, say("Noted, sir."), say("Of course, sir."))
    assert brain.respond("don't forget my exam is on Friday", "s") == brain.MEMORY_FAILED


def test_question_about_memory_is_not_forced_to_save(monkeypatch):
    seen = script(monkeypatch, say("You prefer tea, sir."))
    assert brain.respond("do you remember what I like to drink?", "s") == "You prefer tea, sir."
    assert len(seen) == 1


def test_promised_action_without_a_tool_call_is_retried(monkeypatch):
    seen = script(monkeypatch, say("I shall adjust the volume, sir."),
                  call("get_current_time"), say("Turned down, sir."))
    reply = brain.respond("It's way too loud", "s")
    assert seen[1][-1]["content"] == brain.ACTION_NUDGE
    assert reply == "I shall adjust the volume, sir. Turned down, sir."  # a space between rounds


def test_promise_after_a_tool_call_does_not_repeat_the_action(monkeypatch):
    seen = script(monkeypatch, call("pc_control", action="volume_down"),
                  say("Apologies, sir. I shall adjust the volume immediately."))
    brain.respond("It's way too loud", "s")
    assert len(seen) == 2  # no second volume_down


@pytest.mark.parametrize("filler", ["I am here, sir.", "I'm here, sir", "Yes, sir?", "At your service, sir.",
                                    "Is there something I can assist you with?",
                                    "I am here, sir. Is there something I can assist you with?"])
def test_presence_filler_alone_becomes_an_acknowledgement(monkeypatch, filler):
    script(monkeypatch, say(filler))
    assert brain.respond("Stop", "s") == brain.ACKNOWLEDGE


def test_promise_is_only_retried_once(monkeypatch):
    seen = script(monkeypatch, say("Let me check that, sir."), say("Let me check that, sir."))
    brain.respond("hmm", "s")
    assert len(seen) == 2


@pytest.mark.parametrize("reply,expected", [
    ("I cannot reach the internet, sir. Would you like me to suggest a reliable site?",
     "I cannot reach the internet, sir."),
    ("Would you like me to suggest something?", "Would you like me to suggest something?"),
    ("Shall I set a reminder for it?", "Shall I set a reminder for it?"),
    ("The volume has been reduced, sir. Let me know if it requires further adjustment.",
     "The volume has been reduced, sir."),
    ("Done, sir. Feel free to ask again.", "Done, sir."),
])
def test_trailing_offers_are_dropped(reply, expected):
    assert brain.strip_stock_offer(reply) == expected


def test_copied_example_line_is_retried_before_it_is_spoken(monkeypatch):
    example = next(b for a, b in config.VOICE_EXAMPLE_PAIRS if brain.copies_example(b.split(". ")[0] + "."))
    seen = script(monkeypatch, say(example), say("Goats, sir. Sturdy animals."))
    events = list(brain.respond_events("Goats.", "s"))
    spoken = [e["text"] for e in events if e["type"] == "sentence"]
    assert spoken == ["Goats, sir.", "Sturdy animals."]
    assert seen[1][-1]["content"] == brain.PARROT_NUDGE.format(message="Goats.")
    assert all(m.get("content") != example for m in seen[1])  # the copy is not kept in history


def test_one_sentence_copy_is_caught_too(monkeypatch):
    seen = script(monkeypatch, say("A gatekeeper for your network, sir."), say("Goats, sir."))
    monkeypatch.setattr(brain, "EXAMPLE_LINES", {brain._words("A gatekeeper for your network, sir.")})
    assert brain.respond("Goats.", "s") == "Goats, sir."
    assert len(seen) == 2


def test_copied_example_twice_is_let_through(monkeypatch):
    example = next(b for a, b in config.VOICE_EXAMPLE_PAIRS if brain.copies_example(b.split(". ")[0] + "."))
    seen = script(monkeypatch, say(example), say(example))
    assert brain.respond("Goats.", "s")
    assert len(seen) == 2


def test_short_acknowledgement_examples_may_be_reused(monkeypatch):
    seen = script(monkeypatch, say("Very good, sir."))
    assert brain.respond("Okay", "s") == "Very good, sir."
    assert len(seen) == 1
