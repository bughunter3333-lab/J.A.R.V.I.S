"""Offline tests for the style checker (the real run needs Ollama)."""
import config
from evals import style_eval


def test_good_reply_passes_everything():
    assert all(style_eval.check("Good morning, sir. You have a clear desk, for once.", "chat").values())


def test_rules_catch_bad_replies():
    c = style_eval.check
    assert not c("Sure! Here you go!", "chat")["no_exclamations"]
    assert not c("- one\n- two", "chat")["no_markdown"]
    assert not c("**Bold** answer", "chat")["no_markdown"]
    assert not c("Nice one 😀", "chat")["no_emoji"]
    assert not c("Yes sir, of course sir.", "chat")["sir_not_overused"]
    assert not c("My favorite color is blue.", "chat")["british_spelling"]
    assert not c("As an AI, I cannot feel tired.", "chat")["not_robotic"]
    assert not c("One. Two. Three. Four.", "chat")["short"]
    assert c("One. Two. Three. Four.", "explain")["short"]
    assert not c("I was built by Microsoft.", "identity")["identity_correct"]
    assert c("I was not made by Microsoft, sir; you built me.", "identity")["identity_correct"]


def test_run_uses_a_throwaway_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "real.db")
    replies = iter(["Good morning, sir."] * 100)
    s = style_eval.run(respond=lambda msg, session: next(replies), show=lambda *_: None)
    assert s["replies"] == len(style_eval.PROMPTS)
    assert config.DB_PATH == tmp_path / "real.db"          # restored afterwards
    assert not (tmp_path / "real.db").exists()               # sir's real database never touched


def test_prompt_sections_are_all_in_the_system_prompt():
    for part in (config.IDENTITY, config.VOICE, config.VOICE_EXAMPLES, config.TOOL_RULES, config.SAFETY_RULES):
        assert part in config.SYSTEM_PROMPT


def _words(text):
    import re
    return set(re.findall(r"[a-z']+", text.lower())) - {"a", "the", "i", "is", "you", "my", "to", "of", "what", "what's"}


def test_voice_examples_stay_off_the_style_test_topics():
    """If an example covers the same thing the test asks, the test measures memorisation."""
    for said, _ in config.VOICE_EXAMPLE_PAIRS:
        for prompt, _ in style_eval.PROMPTS:
            a, b = _words(said), _words(prompt)
            overlap = len(a & b) / max(1, len(a | b))
            assert overlap < 0.34, (said, prompt)


def test_new_rules():
    assert not style_eval.check("I'm here, sir.", "short")["no_presence_filler"]
    assert not style_eval.check("Let me look that up for you, sir.", "chat")["no_false_promise"]
    assert style_eval.check("I'm a language model running on your PC, sir.", "identity")["not_robotic"]
    assert not style_eval.check("As a language model I can't.", "chat")["not_robotic"]
    copied = config.VOICE_EXAMPLE_PAIRS[0][1]
    assert not style_eval.check(copied, "chat")["not_parroting"]
    assert style_eval.check("Stopped, sir.", "short")["not_parroting"]


def test_copied_example_opener_is_parroting(monkeypatch):
    from evals import style_eval
    monkeypatch.setattr(style_eval.config, "VOICE_EXAMPLE_PAIRS",
                        (("It's raining again.", "So it is, sir. Ideal for staying in."), ("Never mind.", "Very good, sir.")))
    assert not style_eval.check("So it is, sir. A curious subject.", "short")["not_parroting"]
    assert style_eval.check("Very good, sir.", "short")["not_parroting"]  # plain acknowledgement is fine


def test_joke_and_unseen_rules():
    assert not style_eval.check("I regret, sir, that I am not equipped to tell jokes.", "joke")["tells_joke_when_asked"]
    assert style_eval.check("I told my firewall a joke, sir. It didn't let it through.", "joke")["tells_joke_when_asked"]
    assert not style_eval.check("It is clean and well structured, sir.", "unseen")["no_opinion_on_unseen"]
    assert style_eval.check("I haven't seen it yet, sir. Show me the file.", "unseen")["no_opinion_on_unseen"]
