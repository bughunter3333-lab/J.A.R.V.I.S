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
CLAIMS_MEMORY = re.compile(r"\b(remembered|committed (it|that) to memory|made a note of)\b", re.I)
# Stock sign-offs the model adds even when told not to. Removed in code.
STOCK_OFFER = re.compile(
    r"\s*(?:Is there|Will there be|Do you need|Would you like|May I|How (?:else )?(?:may|can) I)"
    r"[^.?!]{0,60}\b(?:anything else|further|other|more|help|assist(?:ance)?)\b[^.?!]*\?\s*$",
    re.I,
)
MEMORY_FAILED = "I am afraid that did not save to memory, sir. Would you tell me once more?"
NUDGE = (
    "You said you remembered something but did not call the remember_fact tool, "
    "so nothing was saved. Call remember_fact now with the fact, then reply to sir."
)


def strip_stock_offer(reply: str) -> str:
    """Drop a trailing 'Is there anything else I can assist you with?' (unless that's all he said)."""
    trimmed = STOCK_OFFER.sub("", reply).strip()
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
        self.buffer, self.sent = "", []

    def _clean(self, sentence: str) -> str | None:
        sentence = security.redact(sentence.strip())
        if self.sent and not STOCK_OFFER.sub("", sentence).strip():
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
    reply, called, nudged = "", set(), False
    for _ in range(config.MAX_TOOL_ROUNDS):
        trace["rounds"] += 1
        text, tool_calls, tool_message = "", [], None
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
                    yield {"type": "sentence", "text": sentence}
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
            and CLAIMS_MEMORY.search(reply)
            and "remember_fact" not in called
        )
        if false_claim and not nudged:
            nudged = True
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": NUDGE})
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

    final = out.reply or security.redact(reply)
    memory.add_message(session, "user", security.clean_for_storage("history", message))
    memory.add_message(session, "assistant", final)
    yield {"type": "done", "reply": final}
