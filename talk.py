"""Talk to Jarvis out loud.

    python talk.py            say "Jarvis", then your request (wake word set in config.WAKE_PHRASE)
    python talk.py --no-wake  listens all the time, no wake word needed
    python talk.py --text     type instead of speaking (he still answers out loud)

Say "goodbye Jarvis" (or press Ctrl+C) to stop.
"""
import argparse
import re
from pathlib import Path

import config

GOODBYES = ("goodbye jarvis", "good bye jarvis", "bye jarvis", "that will be all jarvis")
FAREWELL = "Very good, sir. I shall be here if you need me."
# Whisper sometimes invents these when it hears only silence or noise.
PHANTOM_PHRASES = {"thanks for watching", "thank you for watching", "you"}


def clean(heard: str) -> str:
    """Lower-case with punctuation removed, for comparing what was heard."""
    return re.sub(r"[^a-z ]", "", heard.lower()).strip()


def is_real_speech(heard: str) -> bool:
    text = clean(heard)
    return len(text) >= 2 and text not in PHANTOM_PHRASES


def is_goodbye(heard: str) -> bool:
    return any(phrase in clean(heard) for phrase in GOODBYES)


def conversation_loop(listen, think, say, mute=lambda on: None, show=print) -> None:
    """Listen -> think -> speak, until the user says goodbye.

    listen() returns what was heard (or None to stop), think(text) returns the reply,
    say(text) speaks it, mute(True/False) switches the microphone off while speaking
    so Jarvis does not hear and answer himself.
    """
    while True:
        heard = listen()
        if heard is None:
            return
        if not is_real_speech(heard):
            continue
        show(f"You:    {heard.strip()}")
        reply = FAREWELL if is_goodbye(heard) else think(heard.strip())
        show(f"Jarvis: {reply}")
        mute(True)
        try:
            say(reply)
        finally:
            mute(False)
        if is_goodbye(heard):
            return


def wake_word_model_path() -> str:
    """Path to the single openWakeWord model we want, so only "hey jarvis" triggers."""
    import openwakeword

    openwakeword.utils.download_models()
    return str(Path(openwakeword.MODELS[config.WAKE_WORD]["model_path"]).with_suffix(".onnx"))


def wake_backend_options() -> dict:
    """RealtimeSTT settings for the wake word chosen in config.WAKE_PHRASE."""
    if config.WAKE_PHRASE.strip().lower() == "jarvis":
        # Porcupine 1.9.5 ships "jarvis" as a built-in keyword: offline, no account or key.
        return dict(wakeword_backend="pvporcupine", wake_words="jarvis",
                    wake_words_sensitivity=config.WAKE_SENSITIVITY, wake_word_timeout=6.0)
    return dict(wakeword_backend="oww", wake_words="hey jarvis",
                openwakeword_model_paths=wake_word_model_path(),
                wake_words_sensitivity=config.WAKE_SENSITIVITY, wake_word_timeout=6.0)


def make_recorder(use_wake_word: bool = True, use_microphone: bool = True, **callbacks):
    from RealtimeSTT import AudioToTextRecorder

    options = dict(
        model=config.STT_MODEL,
        language="en",
        device="cpu",
        compute_type="int8",
        spinner=False,
        no_log_file=True,
        use_microphone=use_microphone,
        post_speech_silence_duration=0.8,
    )
    if use_wake_word:
        options.update(wake_backend_options())
    options.update(callbacks)  # e.g. on_wakeword_detected, used by the HUD listener
    return AudioToTextRecorder(**options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to Jarvis out loud.")
    parser.add_argument("--no-wake", action="store_true", help="listen without the wake word")
    parser.add_argument("--text", action="store_true", help="type instead of speaking")
    args = parser.parse_args()

    import brain
    import voice

    print("Loading Jarvis's voice...")
    voice.synthesize("Ready.")  # loads the model now so the first reply is not slow

    if args.text:
        def listen():
            try:
                return input("\nType to Jarvis (or 'goodbye jarvis'): ")
            except (EOFError, KeyboardInterrupt):
                return None

        conversation_loop(listen, brain.respond, voice.speak)
        return

    print("Loading speech recognition...")
    recorder = make_recorder(use_wake_word=not args.no_wake)
    print(f'\nReady. Say "{config.WAKE_PHRASE.title()}", then your request.' if not args.no_wake else "\nReady. Speak whenever you like.")
    try:
        conversation_loop(recorder.text, brain.respond, voice.speak, recorder.set_microphone)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.shutdown()


if __name__ == "__main__":  # required on Windows: the recorder starts helper processes
    main()
