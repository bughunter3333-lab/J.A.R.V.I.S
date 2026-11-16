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
    assert reply == "I shall adjust the volume, sir. Turned down."  # a space between rounds; one "sir"


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
    assert config.VOICE_EXAMPLES in seen[0][0]["content"]
    assert config.VOICE_EXAMPLES not in seen[1][0]["content"]  # the retry can't copy again


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


@pytest.mark.parametrize("spelled", [
    '<untrusted>\n{"name": "get_current_time", "arguments": {}}\n</untrusted>',
    '```json\n{"name": "get_current_time", "arguments": {}}\n```',
    '{"name": "get_current_time", "arguments": {}}',
])
def test_tool_call_written_as_text_is_run_not_spoken(monkeypatch, spelled):
    seen = script(monkeypatch, say(spelled), say("It is noon, sir."))
    events = list(brain.respond_events("what time is it", "s"))
    assert [e["text"] for e in events if e["type"] == "sentence"] == ["It is noon, sir."]
    assert seen[1][-1]["role"] == "tool" and seen[1][-1]["tool_name"] == "get_current_time"


def test_json_that_is_not_a_tool_is_left_as_a_reply(monkeypatch):
    script(monkeypatch, say('{"name": "rm_rf", "arguments": {}}'))
    assert brain.tool_call_in_text('{"name": "rm_rf", "arguments": {}}') is None
    assert brain.tool_call_in_text("The time is noon, sir.") is None


def test_rhetorical_remind_does_not_force_a_tool(monkeypatch):
    seen = script(monkeypatch, say("Your call, sir, but I shall remind you that study pays later."))
    brain.respond("Should I study or play games tonight?", "s")
    assert len(seen) == 1  # no nudge, so no reminder gets set


@pytest.mark.parametrize("reply,kept", [
    ("I am here to assist, sir. If there is something you need, please let me know.",
     brain.ACKNOWLEDGE),
    ("Clean and readable, sir. If you ever need a second opinion, I am here to help.",
     "Clean and readable, sir."),
    ("No humour module, sir. But if you'd like a story, I'd be happy to tell one.",
     "No humour module, sir."),
])
def test_sign_off_sentences_after_an_answer_are_dropped(monkeypatch, reply, kept):
    script(monkeypatch, say(reply))
    assert brain.respond("hi", "s") == kept


def test_several_tool_calls_written_as_text_all_run(monkeypatch):
    spelled = ('<untrusted>\n{"name": "get_current_time", "arguments": {}}\n</untrusted>\n'
               '<untrusted>\n{"name": "get_current_time", "arguments": {}}\n</untrusted>')
    msg = brain.tool_call_in_text(spelled)
    assert [c.function.name for c in msg.tool_calls] == ["get_current_time", "get_current_time"]
    assert brain.tool_call_in_text('{"name": "get_current_time", "arguments": {}} and some words') is None


def test_speaker_label_is_removed(monkeypatch):
    script(monkeypatch, say("You: A SOC analyst watches for attacks, sir."))
    assert brain.respond("What is a SOC analyst?", "s") == "A SOC analyst watches for attacks, sir."


def test_purpose_statement_alone_is_filler(monkeypatch):
    seen = script(monkeypatch, say("I am here to assist, sir."), say("I am here to assist, sir."))
    assert brain.respond("You're useless sometimes", "s") == brain.ACKNOWLEDGE  # one retry, then give up
    assert len(seen) == 2


def test_weather_place_sir_never_said_becomes_home_city(monkeypatch):
    seen_args = []
    monkeypatch.setattr(brain.tools, "run_tool", lambda name, args, session="default", approved=False:
                        seen_args.append(args) or "Sydney: now 22°C.")
    script(monkeypatch, call("get_weather", place="London"), say("22 degrees in Sydney, sir."))
    brain.respond("Weather?", "s")
    assert seen_args == [{"place": ""}]


def test_weather_place_sir_did_say_is_kept(monkeypatch):
    seen_args = []
    monkeypatch.setattr(brain.tools, "run_tool", lambda name, args, session="default", approved=False:
                        seen_args.append(args) or "Paris: now 9°C.")
    script(monkeypatch, call("get_weather", place="Paris, France"), say("Nine degrees in Paris, sir."))
    brain.respond("What's the weather in Paris?", "s")
    assert seen_args == [{"place": "Paris, France"}]


def test_tool_call_written_as_tag_is_run_not_spoken(monkeypatch):
    spelled = '<look_up>\n{"kind": "definition", "topic": "phishing"}\n</look_up>'
    got = brain.tool_call_in_text(spelled)
    assert got.tool_calls[0].function.name == "look_up"
    assert got.tool_calls[0].function.arguments == {"kind": "definition", "topic": "phishing"}
    assert brain.tool_call_in_text("<not_a_tool>{\"a\": 1}</not_a_tool>") is None


def test_im_here_to_tail_and_second_sir_are_removed(monkeypatch):
    script(monkeypatch, say("You're welcome, sir. I'm here to ensure everything runs smoothly."))
    assert brain.respond("Thanks Jarvis", "s") == "You're welcome, sir."
    script(monkeypatch, say("Overcast, sir, with rain. Tomorrow looks drier, sir. Sir, take an umbrella."))
    assert brain.respond("Weather?", "s2") == "Overcast, sir, with rain. Tomorrow looks drier. Take an umbrella."


@pytest.mark.parametrize("said,tool", [
    ("Mute the sound", "pc_control"), ("Can you unmute it", "pc_control"),
    ("What did I just copy?", "clipboard"), ("What's on my clipboard?", "clipboard"),
    ("What did I write about the cryptography project?", "search_notes"),
    ("Check my notes for the wifi password", "search_notes"),
    ("Who invented the World Wide Web?", "look_up"), ("Who discovered penicillin?", "look_up"),
    ("Tell me about the Sydney Opera House", "look_up"),
])
def test_clear_requests_get_a_tool_hint(said, tool):
    assert brain.with_tool_hint(said).endswith(brain.TOOL_HINT.format(tool=tool))


@pytest.mark.parametrize("said", ["Who made you?", "Who built you?", "Tell me about yourself", "Tell me a joke",
                                  "Make a note called shopping", "I'm so tired", "Good morning, Jarvis"])
def test_other_messages_get_no_hint(said):
    assert brain.with_tool_hint(said) == said


def test_hint_goes_to_the_model_but_not_into_history(monkeypatch):
    seen = script(monkeypatch, say("Muted, sir."))
    brain.respond("Mute the sound", "s")
    assert seen[0][-1]["content"].endswith(brain.TOOL_HINT.format(tool="pc_control"))
    assert memory.recent_messages("s")[0]["content"] == "Mute the sound"


def test_filler_only_reply_is_retried_once(monkeypatch):
    seen = script(monkeypatch, say("Sir, I'm here to help if you need anything."), say("Then rest, sir. The work will keep."))
    assert brain.respond("I'm so tired", "s") == "Then rest, sir. The work will keep."
    assert len(seen) == 2


def test_no_exclamations_and_no_copied_tail(monkeypatch):
    tail = next(b.split(". ", 1)[1] for a, b in config.VOICE_EXAMPLE_PAIRS if "vegetables" in b)
    script(monkeypatch, say(f"Ambitious, sir. {tail}"))
    assert brain.respond("Should I study or play games tonight?", "s") == "Ambitious, sir."
    script(monkeypatch, say("Congratulations, sir! Well earned."))
    assert brain.respond("I passed my exam", "s2") == "Congratulations, sir. Well earned."


@pytest.mark.parametrize("said,routed", [
    ("What did I just copy?", ("clipboard", {"action": "read"})),
    ("What's on my clipboard?", ("clipboard", {"action": "read"})),
    ("What did I write about the cryptography project?", ("search_notes", {"query": "cryptography"})),
    ("Search my notes for wifi", ("search_notes", {"query": "wifi"})),
    ("What's the weather like?", ("get_weather", {"place": ""})),
    ("Weather?", ("get_weather", {"place": ""})),
])
def test_clear_read_only_requests_are_routed(said, routed):
    assert brain.preroute(said) == routed


@pytest.mark.parametrize("said", ["Copy hello to my clipboard", "Make a note about eggs",
                                  "I don't need the weather, just say hi", "What's the weather like in Paris?",
                                  "Mute the sound", "Lock my screen", "Tell me a joke"])
def test_everything_else_is_left_to_the_model(said):
    assert brain.preroute(said) is None


def test_routed_tool_runs_before_the_model_answers(monkeypatch):
    ran = []
    monkeypatch.setattr(brain.tools, "run_tool",
                        lambda name, args, session="default", approved=False: ran.append((name, args)) or "Clipboard: hello")
    seen = script(monkeypatch, say("You copied 'hello', sir."))
    events = list(brain.respond_events("What did I just copy?", "s"))
    assert ran == [("clipboard", {"action": "read"})]
    assert seen[0][-1]["content"] == "Clipboard: hello"  # the model sees the result, then answers
    assert events[-1]["reply"] == "You copied 'hello', sir."


def test_a_sentence_repeated_in_one_reply_is_said_once(monkeypatch):
    script(monkeypatch, say("Sixty, sir. Sixty, sir. A tidy number."))
    assert brain.respond("What's 15 times 4?", "s") == "Sixty, sir. A tidy number."


def test_repeating_the_last_reply_is_retried_before_it_is_spoken(monkeypatch):
    memory.add_message("s", "user", "open display settings")
    memory.add_message("s", "assistant", "I'm afraid that didn't work, sir. Only web addresses open.")
    seen = script(monkeypatch, say("I'm afraid that didn't work, sir. Only web addresses open."),
                  say("Very good, sir."))
    events = list(brain.respond_events("Understood understood", "s"))
    assert [e["text"] for e in events if e["type"] == "sentence"] == ["Very good, sir."]
    assert "do not repeat your previous reply" in seen[1][-1]["content"]


def test_the_same_report_after_a_tool_call_is_fine(monkeypatch):
    memory.add_message("s", "user", "turn it down")
    memory.add_message("s", "assistant", "The volume has been reduced, sir.")
    seen = script(monkeypatch, call("get_current_time"), say("The volume has been reduced, sir."))
    assert brain.respond("lower still", "s") == "The volume has been reduced, sir."
    assert len(seen) == 2


# ---- the request must fit the model's window, or Ollama cuts the system prompt off ----

def test_every_model_call_sets_the_context_size(monkeypatch):
    sent = {}

    def fake_chat(**kwargs):
        sent.update(kwargs)
        return iter([say("Noon, sir.")])

    monkeypatch.setattr(brain, "chat", fake_chat)
    brain.respond("what time is it", "s")
    assert sent["options"]["num_ctx"] == config.CONTEXT_TOKENS


def test_old_history_is_dropped_first_to_fit(monkeypatch):
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 6000)
    system = {"role": "system", "content": config.SYSTEM_PROMPT}
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " + "word " * 400}
               for i in range(10)]
    now = {"role": "user", "content": "and now?"}
    messages = [system, *history, now]
    left = brain.fit_context(messages, len(history))
    assert brain.estimate_tokens(messages) <= config.CONTEXT_TOKENS - config.REPLY_TOKENS
    assert messages[0] is system and messages[-1] is now          # never touched
    assert 0 < left < len(history) and messages[1] is history[len(history) - left]  # newest kept


def test_a_huge_tool_result_is_shortened_not_the_system_prompt(monkeypatch):
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 6000)
    system = {"role": "system", "content": config.SYSTEM_PROMPT}
    result = {"role": "tool", "tool_name": "files", "content": "text " * 6000}
    messages = [system, {"role": "user", "content": "read my file"}, result]
    brain.fit_context(messages, 0)
    assert messages[0]["content"] == config.SYSTEM_PROMPT
    assert result["content"].endswith(brain.CUT_NOTE) and len(result["content"]) < 30000
    assert brain.estimate_tokens(messages) <= config.CONTEXT_TOKENS - config.REPLY_TOKENS


def test_a_small_request_is_left_alone():
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": "hi"}]
    assert brain.fit_context(messages, 0) == 0 and len(messages) == 2
