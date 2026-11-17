"""Instant commands: simple, safe requests answered without waiting for the big model.

A small nearest-neighbour classifier on the embedding model (semantic.py): sir's message
is compared, by meaning, with example phrasings of each command below, and with
"not a command" sentences that share their words ("the volume of a sphere", "next week").
Only a message that is clearly closest to one command, and clearly far from everything
else, is handled here; anything unsure goes to the model as before.

Only harmless actions run instantly (volume, music, the time). Locking the screen goes
through tools.run_tool like any call, so it still waits for sir's approval.

The example phrasings must never be the test questions in evals/ (tests check this),
or the scores would measure memory instead of understanding.
"""
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Intent:
    tool: str
    args: dict
    examples: tuple[str, ...]


INTENTS = {
    "volume_up": Intent("pc_control", {"action": "volume_up"}, (
        "louder please", "make it louder", "raise the volume", "crank it up", "volume up",
        "it's too quiet", "I can barely hear the music", "bump the sound up", "increase the sound",
        "a bit louder", "the sound is too low", "turn the speakers up")),
    "volume_down": Intent("pc_control", {"action": "volume_down"}, (
        "quieter please", "make it quieter", "lower the volume", "volume down", "turn the sound down",
        "that's way too noisy", "it's deafening", "bring the volume down a bit", "decrease the sound",
        "the music is too loud", "less loud please", "my ears hurt, turn it down")),
    "mute": Intent("pc_control", {"action": "mute"}, (
        "mute", "mute the speakers", "silence the speakers", "kill the sound", "no sound please",
        "unmute", "unmute the speakers", "switch the sound back on", "sound off")),
    "play_pause": Intent("pc_control", {"action": "play_pause"}, (
        "pause", "pause it", "stop the music", "hold the song", "resume", "resume the music",
        "carry on playing", "play the music again", "pause the video", "unpause")),
    "next_track": Intent("pc_control", {"action": "next_track"}, (
        "next track", "skip it", "skip this one", "play the next song", "I don't like this song, skip",
        "move on to the next tune", "next one please", "change the song")),
    "previous_track": Intent("pc_control", {"action": "previous_track"}, (
        "previous track", "play that last song again", "back one song", "go back a track",
        "replay the song before this", "the one before please", "rewind to the last tune")),
    "lock_screen": Intent("pc_control", {"action": "lock_screen"}, (
        "lock the computer", "lock my PC", "lock it up", "lock the screen please", "I'm heading out, lock up",
        "lock the workstation", "secure my PC while I'm away", "nobody should use the computer, lock it")),
    "current_time": Intent("get_current_time", {}, (
        "what's the time", "time please", "what time have we got", "tell me the time", "the time, Jarvis",
        "what o'clock is it", "do you have the time", "current time")),
}

# Sentences that share words with the commands but are not commands. A message closer to
# one of these than to any command is left to the model.
NOT_COMMANDS = (
    "what's the volume of a sphere", "the volume of this book is huge", "next week I have an exam",
    "what's next on my schedule", "I paused my studies last year", "lock in and focus",
    "time flies when you're having fun", "what time does the match start", "how long until midnight",
    "set a timer for ten minutes", "remind me in five minutes", "how's the weather looking",
    "the music at the party was great", "who sings this song", "what song is this",
    "play me a funny joke", "turn up at the party", "my neighbour is so loud", "I can't hear you properly",
    "speak up, Jarvis", "say that again", "stop talking", "that's enough, thanks", "good morning",
    "tell me something funny", "how are you", "what's the date today", "which day of the week is it", "is the music too loud for you",
    "search the web for new songs", "open spotify", "open the youtube website", "go back to what we were discussing",
    "skip the small talk", "previous conversation", "quiet day today", "it's quiet in here",
    "secure my accounts", "how do I lock a file", "what's my password manager",
)

REPLIES = {
    "volume_up": "Louder, sir.", "volume_down": "Quieter, sir.", "mute": "Done, sir.",
    "play_pause": "Done, sir.", "next_track": "Next track, sir.", "previous_track": "Back one, sir.",
}


def classify(message: str) -> tuple[str, float, float] | None:
    """(intent, score, margin) if the message is clearly one instant command, else None."""
    import semantic

    text = message.strip()
    if not text or len(text.split()) > 12:
        return None  # long messages carry more than a command: let the model read them
    labels, examples = [], []
    for name, intent in INTENTS.items():
        for ex in intent.examples:
            labels.append(name)
            examples.append(ex)
    labels += ["none"] * len(NOT_COMMANDS)
    examples += list(NOT_COMMANDS)
    try:
        ranked = semantic.rank(text, examples)
    except semantic.Unavailable:
        return None
    best_by_label: dict[str, float] = {}
    for score, i in ranked:
        best_by_label.setdefault(labels[i], score)
    top_label, top = max(best_by_label.items(), key=lambda kv: kv[1])
    if top_label == "none":
        return None
    runner_up = max(score for label, score in best_by_label.items() if label != top_label)
    margin = top - runner_up
    if top < config.INSTANT_MIN_SIMILARITY or margin < config.INSTANT_MARGIN:
        return None
    return top_label, top, margin


def reply_for(intent: str, result: str) -> str:
    """What he says after an instant command ran."""
    if intent == "current_time":
        clock = result.rsplit(", ", 1)[-1].lstrip("0") if ", " in result else result
        return f"It's {clock}, sir."
    if result.startswith("WAITING FOR APPROVAL"):
        return "Shall I lock the screen, sir?"
    if result.startswith(("Refused", "Failed")):
        return f"I'm afraid I couldn't, sir. {result.split(':', 1)[-1].strip()}"
    return REPLIES.get(intent, "Done, sir.")
