"""Memory by meaning, with a fake embedding model (no Ollama needed)."""
import re

import pytest

import brain
import config
import memory
import semantic
import tools
from tests.test_brain import say, script

TOPICS = {"exam": ["exam", "test", "quiz"], "drink": ["tea", "coffee", "drink"],
          "allergy": ["peanut", "peanuts", "satay", "allergic"], "car": ["car", "toyota", "drive"]}


def fake_embed(texts):
    """A word-topic vector: texts about the same topic point the same way."""
    out = []
    for t in texts:
        words = re.findall(r"[a-z]+", t.lower())
        out.append([1.0 + sum(w in syn for w in words) * 5 for syn in TOPICS.values()])
    return out


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "workspace")
    memory.init_db()


@pytest.fixture
def embedder(monkeypatch):
    calls = []
    monkeypatch.setattr(semantic, "_embed", lambda texts: calls.append(list(texts)) or fake_embed(texts))
    return calls


FACTS = ["Sir's exam is on Friday.", "Sir prefers tea over coffee.", "Sir is allergic to peanuts.",
         "Sir's car is a Toyota.", "Sir lives in Sydney.", "Sir supports Barcelona.", "Sir has a ThinkPad.",
         "Sir's sister is Mim.", "Sir's gym days are Monday and Thursday.", "Sir's mate is Rafi."]


def test_few_facts_all_go_without_embedding(embedder):
    assert semantic.relevant_facts("anything", FACTS[:3]) == FACTS[:3] and embedder == []


def test_many_facts_only_related_ones_plus_newest(embedder, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_TOP_K", 1)
    chosen = semantic.relevant_facts("can I eat this satay?", FACTS)
    assert "Sir is allergic to peanuts." in chosen
    assert "Sir's car is a Toyota." not in chosen
    assert FACTS[-2:] == chosen[-2:]  # the newest are kept: he may have just been told


def test_vectors_are_cached(embedder):
    semantic.vectors(FACTS)
    semantic.vectors(FACTS)
    assert sum(len(c) for c in embedder) == len(FACTS)


def test_no_embedding_model_falls_back_to_all_facts(monkeypatch):
    def broken(texts):
        raise semantic.Unavailable("model not found")
    monkeypatch.setattr(semantic, "_embed", broken)
    assert semantic.relevant_facts("satay?", FACTS) == FACTS


def test_only_related_facts_reach_the_model(embedder, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_TOP_K", 1)
    for f in FACTS:
        memory.add_fact(f)
    seen = script(monkeypatch, say("Best not, sir: peanuts."))
    brain.respond("can I eat this satay?", "s")
    sent = seen[0][-1]["content"]
    assert "allergic to peanuts" in sent and "Toyota" not in sent
    assert seen[0][0]["content"] == config.SYSTEM_PROMPT


def test_notes_are_found_by_meaning(embedder):
    config.WORKSPACE.mkdir(parents=True)
    (config.WORKSPACE / "friday.txt").write_text("Revise chapters 3-5 before the exam.", encoding="utf-8")
    (config.WORKSPACE / "shopping.txt").write_text("Milk, bread, eggs.", encoding="utf-8")
    out = tools.run_tool("search_notes", {"query": "my test prep"})
    assert "friday.txt" in out and "related" in out and "shopping.txt" not in out


def test_exact_note_matches_still_come_first(embedder):
    config.WORKSPACE.mkdir(parents=True)
    (config.WORKSPACE / "shopping.txt").write_text("Milk, bread, eggs.", encoding="utf-8")
    out = tools.run_tool("search_notes", {"query": "bread"})
    assert "shopping.txt: Milk, bread, eggs." in out and "related" not in out
