"""Shared test setup."""
import pytest

import semantic


@pytest.fixture(autouse=True)
def no_real_embedding_model(monkeypatch):
    """Tests never reach a real Ollama for embeddings: results would depend on whether it is
    running on this machine. Tests that need meaning bring their own fake (see test_semantic)."""
    def unavailable(texts):
        raise semantic.Unavailable("no embedding model in tests")
    monkeypatch.setattr(semantic, "_embed", unavailable)
