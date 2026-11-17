"""Game mode: give the GPU's memory back for gaming.

    "Jarvis, game mode"        -> unloads every model Ollama has in GPU memory
    "Jarvis, I'm done gaming"  -> loads the main model back so replies are quick again

Normally the models stay loaded for config.MODEL_KEEP_ALIVE (60 minutes) so he answers
fast; on an 8 GB card that leaves a game about 1.5 GB. In game mode he still answers if
spoken to, but every model is unloaded the moment it has finished (keep_alive 0), and
the server doesn't warm the model up at start. The switch is remembered across restarts
(data/game_mode.json).

The phrases are recognised in code (not by the model), so switching game mode on never
loads the 6 GB model just to understand the request.
"""
import json
import re
import threading

import config

ON = re.compile(r"^\s*(?:(?:hey|ok|okay)\s+)?(?:jarvis[\s,.!]*)?(?:(?:switch|turn|go)\s+(?:on\s+|into\s+|to\s+)?)?"
                r"(?:(?:game|gaming)\s+mode(?:\s+on)?|i'?m\s+(?:going\s+)?(?:to\s+)?(?:play(?:ing)?\s+(?:a\s+)?games?|gaming)"
                r"|free\s+(?:up\s+)?(?:the\s+)?(?:gpu|graphics(?:\s+card)?|vram)(?:\s+memory)?)"
                r"(?:[\s,]+(?:please|now|sir|jarvis))*[\s.!]*$", re.I)
OFF = re.compile(r"^\s*(?:(?:hey|ok|okay)\s+)?(?:jarvis[\s,.!]*)?(?:(?:i'?m\s+)?(?:done|finished)\s+(?:with\s+)?(?:gaming|playing|the\s+game)"
                 r"|(?:game|gaming)\s+mode\s+off|(?:switch|turn)\s+off\s+(?:game|gaming)\s+mode|(?:stop|end|exit)\s+(?:game|gaming)\s+mode)"
                 r"(?:[\s,]+(?:please|now|sir|jarvis))*[\s.!]*$", re.I)


def _state_path():
    return config.DATA_DIR / "game_mode.json"


def active() -> bool:
    try:
        return bool(json.loads(_state_path().read_text(encoding="utf-8")).get("on"))
    except (OSError, ValueError):
        return False


def _set(on: bool) -> None:
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps({"on": on}), encoding="utf-8")


def keep_alive():
    """What every model call passes as keep_alive: unload at once in game mode."""
    return 0 if active() else config.MODEL_KEEP_ALIVE


def unload_all() -> list[str]:
    """Unload every model Ollama has loaded. Returns their names."""
    import ollama

    names = []
    for m in ollama.ps().models:
        try:
            ollama.generate(model=m.model, prompt="", keep_alive=0)
            names.append(m.model)
        except Exception:
            pass
    return names


def switch(on: bool) -> str:
    """Turn game mode on or off and say what happened."""
    _set(on)
    if on:
        try:
            freed = unload_all()
        except Exception:
            return "Game mode, sir, though I couldn't reach Ollama to clear the graphics memory."
        what = "the graphics memory is yours" if freed else "the graphics memory was already clear"
        return f"Game mode, sir: {what}. I'll stay out of the way until you're done."
    threading.Thread(target=_warm_up, daemon=True).start()  # answer now, load in the background
    return "Welcome back, sir. Game mode is off; I'm loading up and will be quick again in a moment."


def _warm_up() -> None:
    try:
        import brain

        brain.warm_up()
    except Exception:
        pass  # the next question loads it anyway


def command(message: str) -> bool | None:
    """True = sir asked for game mode, False = asked to end it, None = neither."""
    if ON.match(message):
        return True
    if OFF.match(message):
        return False
    return None
