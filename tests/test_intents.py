"""Instant commands, with a fake embedding model."""
import re

import pytest

import brain
import config
import intents
import memory
import pc
import semantic
from evals import conversation_eval, style_eval, tool_eval
from tests.test_brain import say, script

# Exact fake vectors: each text points along one axis, so similarities are exactly 1 or 0
# unless a text is listed with a mix.
AXES = {"louder": 0, "up": 0, "sphere": 3}


def fake_embed(texts):
    vecs = []
    for t in texts:
        words = re.findall(r"[a-z']+", t.lower().replace("search_query: ", "").replace("search_document: ", ""))
        v = [0.0] * 5
        for w in words:
            if w in AXES:
                v[AXES[w]] += 1.0
        v[4] = 0.01  # never a zero vector
        vecs.append(v)
    return vecs


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    memory.init_db()
    monkeypatch.setattr(semantic, "_embed", fake_embed)


@pytest.fixture
def pressed(monkeypatch):
    keys = []
    monkeypatch.setattr(pc, "control", lambda action, url="", steps=5: keys.append(action) or "Volume up 10%.")
    return keys


def test_a_clear_command_runs_without_the_model(pressed, monkeypatch):
    monkeypatch.setattr(intents, "classify", lambda m: ("volume_up", 0.9, 0.2))
    seen = script(monkeypatch)  # no responses scripted: the model must not be called
    assert brain.respond("make it louder", "s") == "Louder, sir."
    assert pressed == ["volume_up"] and seen == []


def test_unsure_messages_go_to_the_model(pressed, monkeypatch):
    monkeypatch.setattr(intents, "classify", lambda m: None)
    seen = script(monkeypatch, say("About 4.19 times the radius cubed, sir."))
    brain.respond("what's the volume of a sphere", "s")
    assert pressed == [] and len(seen) == 1


def test_the_time_is_said_naturally():
    assert intents.reply_for("current_time", "Monday 16 November 2026, 09:05 PM") == "It's 9:05 PM, sir."


def test_locking_still_asks_first():
    assert intents.reply_for("lock_screen", "WAITING FOR APPROVAL: ...") == "Shall I lock the screen, sir?"


def test_switched_off_means_the_model_answers(pressed, monkeypatch):
    monkeypatch.setattr(config, "INSTANT_COMMANDS", False)
    seen = script(monkeypatch, say("Louder, sir."))
    brain.respond("make it louder", "s")
    assert len(seen) == 1


def test_no_embedding_model_means_the_model_answers(pressed, monkeypatch):
    def broken(texts):
        raise semantic.Unavailable("x")
    monkeypatch.setattr(semantic, "_embed", broken)
    seen = script(monkeypatch, say("Louder, sir."))
    brain.respond("make it louder", "s")
    assert len(seen) == 1


def test_long_messages_are_never_instant():
    assert intents.classify("could you please make it a little louder because I really cannot hear the film") is None


def test_examples_never_copy_the_test_questions():
    """Training on the test would make the eval scores meaningless."""
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()
    tested = {norm(c[0]) for c in tool_eval.CASES} | {norm(p) for p, _ in style_eval.PROMPTS}
    tested |= {norm(t["say"]) for _, turns in conversation_eval.SCENARIOS for t in turns if t != "NEW CHAT"}
    examples = {norm(e) for i in intents.INTENTS.values() for e in i.examples} | {norm(e) for e in intents.NOT_COMMANDS}
    assert not examples & tested, examples & tested


def test_only_harmless_or_approved_actions_are_instant():
    import approvals
    for name, intent in intents.INTENTS.items():
        assert intent.tool in ("pc_control", "get_current_time")
        if intent.args.get("action") == "lock_screen":
            assert approvals.needs_approval(intent.tool, intent.args)


def test_classifier_needs_a_clear_lead(monkeypatch):
    monkeypatch.setattr(intents, "INTENTS", {"volume_up": intents.Intent("pc_control", {"action": "volume_up"}, ("louder",))})
    monkeypatch.setattr(intents, "NOT_COMMANDS", ("the volume of a sphere",))
    assert intents.classify("louder")[0] == "volume_up"
    assert intents.classify("sphere") is None            # closest to a non-command
    assert intents.classify("pizza") is None             # close to nothing
