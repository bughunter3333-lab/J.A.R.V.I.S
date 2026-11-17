import types

import pytest

import brain
import config
import gamemode


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.mark.parametrize("text", [
    "game mode", "Jarvis, game mode", "gaming mode on", "switch to game mode please",
    "turn on game mode", "I'm going to play games", "I'm gaming now", "free up the GPU memory",
    "Hey Jarvis, game mode.",
])
def test_recognises_game_mode_on(text):
    assert gamemode.command(text) is True


@pytest.mark.parametrize("text", [
    "I'm done gaming", "Jarvis, I'm done gaming", "done playing", "game mode off",
    "turn off game mode", "stop game mode", "I'm finished with the game",
])
def test_recognises_game_mode_off(text):
    assert gamemode.command(text) is False


@pytest.mark.parametrize("text", [
    "what is game mode in Windows?", "recommend a game", "tell me about my GPU",
    "open the game folder", "how much memory does my GPU have",
])
def test_ignores_other_sentences(text):
    assert gamemode.command(text) is None


def test_switch_on_unloads_and_persists(monkeypatch):
    unloaded = []
    fake = types.SimpleNamespace(
        ps=lambda: types.SimpleNamespace(models=[types.SimpleNamespace(model="qwen3:8b"),
                                                 types.SimpleNamespace(model="nomic-embed-text")]),
        generate=lambda model, prompt, keep_alive: unloaded.append((model, keep_alive)),
    )
    monkeypatch.setitem(__import__("sys").modules, "ollama", fake)
    reply = gamemode.switch(True)
    assert "graphics memory is yours" in reply
    assert unloaded == [("qwen3:8b", 0), ("nomic-embed-text", 0)]
    assert gamemode.active() and gamemode.keep_alive() == 0


def test_switch_off_warms_up(monkeypatch):
    gamemode._set(True)
    calls = []
    done = __import__("threading").Event()
    monkeypatch.setattr(brain, "warm_up", lambda: (calls.append(1), done.set()))
    reply = gamemode.switch(False)
    assert done.wait(5) and calls == [1] and "Game mode is off" in reply
    assert not gamemode.active() and gamemode.keep_alive() == config.MODEL_KEEP_ALIVE


def test_unreadable_state_means_off(state_dir):
    (state_dir / "game_mode.json").write_text("not json")
    assert gamemode.active() is False


def test_brain_handles_it_without_the_model(monkeypatch):
    monkeypatch.setattr(gamemode, "unload_all", lambda: [])
    def no_model(*a, **k):
        raise AssertionError("the model must not be called")
    monkeypatch.setattr(brain, "chat", no_model)
    events = list(brain.respond_events("Jarvis, game mode", "s-game"))
    assert events[-1]["type"] == "done"
    assert "Game mode" in events[-1]["reply"]
    assert gamemode.active()
