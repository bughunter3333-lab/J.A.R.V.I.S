"""The thinking loop: memory + model + tools."""
import re
import time

from ollama import chat

import approvals
import config
import memory
import security
import tools

# Small models sometimes say "I have remembered that" without calling the tool.
# This is checked in code so a false claim never reaches the user.
ASKED_TO_REMEMBER = re.compile(r"\b(remember|keep (it |that |this )?in mind|don'?t forget)\b", re.I)
# An explicit "remember this" (not a question like "do you remember...?"). If sir says
# this and the model doesn't call remember_fact, code makes it try again.
REMEMBER_REQUEST = re.compile(
    r"^(?!\s*(?:do|did|can|could|what|when|where|who|how|why)\b).*\b(?:remember (?:that|this|my|me|to)|"
    r"keep (?:it |that |this )?in mind|don'?t forget|make a note that|note that)\b", re.I)
CLAIMS_MEMORY = re.compile(r"\b(remembered|committed (it|that) to memory|made a note of)\b", re.I)
# Stock sign-offs the model adds even when told not to. Removed in code.
STOCK_OFFER = re.compile(
    r"\s*(?:Is there|Will there be|Do you need|Would you like|May I|How (?:else )?(?:may|can) I)"
    r"[^.?!]{0,60}\b(?:anything else|further|other|more|help|assist(?:ance)?)\b[^.?!]*\?\s*$",
    re.I,
)
# "I shall adjust the volume, sir." with no tool call means nothing happened. Code
# catches the promise and makes the model actually call the tool (once).
PROMISES_ACTION = re.compile(
    r"\b(?:I(?:'ll| will| shall| am going to| am)|let me|allow me)\s+(?:now\s+|just\s+)?"
    r"(?:adjust|adjusting|turn|turning|lower|lowering|raise|raising|set|setting|open|opening|lock|locking|"
    r"check|checking|look|looking|find|finding|search|searching|copy|copying|play|pause|mute|muting|skip|"
    r"remind|convert|fetch|fetching|get|getting|pull|pulling)\b",
    re.I,
)
ACTION_NUDGE = (
    "You told sir you would do that, but you did not call any tool, so nothing happened. "
    "Call the right tool now, then tell sir the result in one short sentence."
)
# "Would you like me to suggest a reliable site?" after a real answer: an offer, dropped.
TRAILING_OFFER = re.compile(
    r"\s*(?:(?:Would you like me to|Shall I|Do you want me to|Should I)\s+"
    r"(?:suggest|recommend|provide|offer|find|look|search|help|try|point)[^.?!]*\?"
    r"|(?:Please |Do )?(?:let me know|feel free to)[^.?!]*[.?!]?)\s*$", re.I)
# qwen3:8b copies a voice example word for word when unsure ("Goats." -> the firewall
# line). Opening sentences of examples for real statements (3+ words said) are checked
# before they are spoken; a copy gets one retry. Short acknowledgements may be reused.
def _words(text: str) -> tuple:
    return tuple(re.findall(r"[a-z']+", text.lower().replace("\u2019", "'")))


EXAMPLE_LINES = {
    _words(line)
    for said, example in config.VOICE_EXAMPLE_PAIRS if len(_words(said)) >= 3
    for line in re.split(r"(?<=[.!?])\s+", example) if len(_words(line)) >= 4
}
PARROT_NUDGE = (
    "(Note from the system, not sir: do not reuse any line from your examples. Reply to sir's "
    "message \"{message}\" in your own words, in one or two short sentences.)"
)


def copies_example(sentence: str) -> bool:
    return _words(sentence) in EXAMPLE_LINES


# A reply that is nothing but presence filler ("I am here, sir.", "Yes, sir?").
PRESENCE_ONLY = re.compile(
    r"^\s*(?:I(?:'m| am) (?:here|listening)|Yes|At your service|How (?:can|may) I (?:help|assist)(?: you)?"
    r"|(?:Is there )?(?:something|anything) (?:I can|you'd like me to) (?:help|assist)(?: you)?(?: with)?)"
    r"(?:,? sir)?[.?!]?\s*$", re.I)
ACKNOWLEDGE = "Very good, sir."
MEMORY_FAILED = "I am afraid that did not save to memory, sir. Would you tell me once more?"
NUDGE = (
    "Sir asked you to remember something, but you did not call the remember_fact tool, "
    "so nothing was saved. Call remember_fact now with the fact, then reply to sir."
)


def strip_stock_offer(reply: str) -> str:
    """Drop a trailing 'Is there anything else I can assist you with?' (unless that's all he said)."""
    trimmed = TRAILING_OFFER.sub("", STOCK_OFFER.sub("", reply)).strip()
    return trimmed or reply


def _system_prompt() -> str:
    facts = memory.all_facts()
    if not facts:
        return config.SYSTEM_PROMPT
    return config.SYSTEM_PROMPT + " Things you know about sir: " + " ".join(
        f"({i}) {fact}" for i, fact in enumerate(facts, 1)
    )


STATUS = {
    "get_current_time": "Checking the time", "open_app": "Opening the app", "create_note": "Saving a note",
    "search_notes": "Searching your notes", "remember_fact": "Committing that to memory",
    "web_search": "Searching the web", "read_webpage": "Reading the page", "get_weather": "Checking the weather",
    "calculate": "Calculating", "system_status": "Checking the system", "reminder": "Setting things up",
    "pc_control": "Working on it", "convert": "Converting", "look_up": "Looking it up",
    "files": "Searching your files", "clipboard": "Checking the clipboard", "about_me": "Checking my records",
}
SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")


def respond(message: str, session: str = "default") -> str:
    """Take what the user said, return Jarvis's whole reply (talk.py and POST /chat use this)."""
    reply = ""
    for event in respond_events(message, session):
        if event["type"] == "done":
            reply = event["reply"]
    return reply


def respond_events(message: str, session: str = "default"):
    """The same thinking loop, as a stream of events for the HUD:

    {"type": "status", "text": "Searching the web"}   a tool is being used
    {"type": "sentence", "text": "..."}                one finished, cleaned sentence
    {"type": "done", "reply": "..."}                   the whole reply, saved to history

    Every request is traced (latency, model rounds, tools used) for GET /stats.
    """
    started = time.perf_counter()
    trace = {"rounds": 0, "tools": []}
    try:
        yield from _respond(message, session, trace)
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        memory.add_trace(session, config.MODEL, elapsed, trace["rounds"], trace["tools"],
                         False, f"{type(exc).__name__}: {exc}")
        raise
    elapsed = (time.perf_counter() - started) * 1000
    memory.add_trace(session, config.MODEL, elapsed, trace["rounds"], trace["tools"], True)


class _Sentences:
    """Collects streamed text and releases it a cleaned sentence at a time."""

    def __init__(self):
        self.buffer, self.sent, self.dropped_filler = "", [], False

    def _clean(self, sentence: str) -> str | None:
        sentence = security.redact(sentence.strip())
        if PRESENCE_ONLY.match(sentence):
            self.dropped_filler = True
            return None  # "I am here, sir." says nothing; never speak it
        if self.sent and not TRAILING_OFFER.sub("", STOCK_OFFER.sub("", sentence)).strip():
            return None  # "Is there anything else...?" after a real answer: drop it
        return sentence or None

    def add(self, text: str):
        self.buffer += text
        while True:
            m = SENTENCE_END.search(self.buffer)
            if not m:
                return
            sentence, self.buffer = self.buffer[:m.end()], self.buffer[m.end():]
            cleaned = self._clean(sentence)
            if cleaned:
                self.sent.append(cleaned)
                yield cleaned

    def flush(self):
        cleaned = self._clean(self.buffer) if self.buffer.strip() else None
        self.buffer = ""
        if cleaned:
            self.sent.append(cleaned)
            yield cleaned

    @property
    def reply(self) -> str:
        return " ".join(self.sent)


EMPTY_REPLY = "Sorry, sir, I didn't quite catch that. Could you say it again?"


def _answer_approval(message: str, session: str, trace: dict) -> str | None:
    """If an action is waiting for sir and this message is his yes/no, settle it in code."""
    waiting = approvals.pending(session)
    if not waiting:
        return None
    verdict = approvals.decision(message)
    if verdict is None:  # he moved on to something else: drop the request
        approvals.settle(waiting["id"], "dropped")
        return None
    if verdict == "deny":
        approvals.settle(waiting["id"], "denied")
        memory.log_tool(waiting["tool"], waiting["args"], "Denied by sir.", False)
        return "Very good, sir. I've left it."
    approvals.settle(waiting["id"], "approved")
    trace["tools"].append(waiting["tool"])
    result = tools.run_tool(waiting["tool"], waiting["args"], session, approved=True)
    if result.startswith(("Refused", "Failed")):
        return f"I'm afraid that didn't work, sir. {result.split(':', 1)[-1].strip()}"
    return f"Done, sir. {result}"


def _respond(message: str, session: str, trace: dict):
    memory.init_db()
    settled = _answer_approval(message, session, trace)
    if settled is not None:
        memory.add_message(session, "user", security.clean_for_storage("history", message))
        memory.add_message(session, "assistant", settled)
        yield {"type": "sentence", "text": settled}
        yield {"type": "done", "reply": settled}
        return

    messages = [{"role": "system", "content": _system_prompt()}]
    messages += memory.recent_messages(session)
    messages.append({"role": "user", "content": message})

    # If sir asked Jarvis to remember something, hold the reply back until the
    # false-memory check below has run, instead of speaking it as it streams.
    hold_back = bool(ASKED_TO_REMEMBER.search(message))
    out = _Sentences()
    reply, called, nudged, pushed_to_act, parrot_retried = "", set(), False, False, False
    for _ in range(config.MAX_TOOL_ROUNDS):
        trace["rounds"] += 1
        text, tool_calls, tool_message, parroted = "", [], None, False
        round_out = _Sentences() if hold_back else out
        for chunk in chat(model=config.MODEL, messages=messages, tools=list(tools.TOOLS.values()),
                          think=False, stream=True):
            piece = chunk.message.content or ""
            text += piece
            if chunk.message.tool_calls:
                tool_calls += list(chunk.message.tool_calls)
                tool_message = chunk.message
            if piece and not tool_calls and not hold_back:
                for sentence in round_out.add(piece):
                    if not parrot_retried and len(round_out.sent) == 1 and copies_example(sentence):
                        parroted = True  # caught before it was spoken
                        break
                    yield {"type": "sentence", "text": sentence}
            if parroted:
                break
        if (not parroted and not parrot_retried and not tool_calls and not hold_back
                and not round_out.sent and copies_example(round_out.buffer)):
            parroted = True  # a one-sentence copy, still unspoken in the buffer
        if parroted:
            parrot_retried = True
            round_out.sent.clear()
            round_out.buffer = ""
            messages.append({"role": "user", "content": PARROT_NUDGE.format(message=message)})
            continue
        reply = text.strip()
        if tool_calls:
            tool_message.content = text
            tool_message.tool_calls = tool_calls
            messages.append(tool_message)
            for call in tool_calls:
                name = call.function.name
                called.add(name)
                trace["tools"].append(name)
                yield {"type": "status", "text": STATUS.get(name, "Working on it")}
                result = tools.run_tool(name, dict(call.function.arguments or {}), session)
                messages.append({"role": "tool", "tool_name": name, "content": result})
            continue
        false_claim = (
            hold_back
            and "remember_fact" not in called
            and (CLAIMS_MEMORY.search(reply) or REMEMBER_REQUEST.search(message))
        )
        if false_claim and not nudged:
            nudged = True
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": NUDGE})
            continue
        if not called and not pushed_to_act and not false_claim and PROMISES_ACTION.search(reply):
            pushed_to_act = True
            if not hold_back:  # finish the spoken promise so the next round starts a new sentence
                for sentence in out.add(" "):
                    yield {"type": "sentence", "text": sentence}
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": ACTION_NUDGE})
            continue
        if false_claim:
            reply = MEMORY_FAILED  # still no tool call: be honest instead of pretending
        break
    else:
        reply = reply or "My apologies, sir. I got tangled in my own tools and gave up."
        hold_back = True  # nothing was streamed for this ending; send it whole

    if hold_back:
        out = _Sentences()
        for sentence in out.add(reply + " "):
            yield {"type": "sentence", "text": sentence}
    for sentence in out.flush():
        yield {"type": "sentence", "text": sentence}

    final = out.reply or ("" if out.dropped_filler else security.redact(reply))
    if out.dropped_filler and not out.sent or PRESENCE_ONLY.match(final):
        # The whole reply was "I am here, sir.": for a short message that means "noted".
        final = ACKNOWLEDGE
        out.sent.append(final)
        yield {"type": "sentence", "text": final}
    if not final.strip():
        # The model sometimes returns nothing (often for a mis-heard fragment). Never
        # leave sir with silence and a "..." on screen.
        final = EMPTY_REPLY
        yield {"type": "sentence", "text": final}
    memory.add_message(session, "user", security.clean_for_storage("history", message))
    memory.add_message(session, "assistant", final)
    yield {"type": "done", "reply": final}
