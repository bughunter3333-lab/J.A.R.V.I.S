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
