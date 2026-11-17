"""The thinking loop: memory + model + tools."""
import json
import re
import time

from ollama import Message, chat

import approvals
import config
import gamemode
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
    # Only harmless or approval-gated actions: never "remind"/"set", which the model uses
    # rhetorically ("I shall remind you that studying pays") and which act without asking.
    r"(?:adjust|adjusting|turn|turning|lower|lowering|raise|raising|open|opening|lock|locking|"
    r"check|checking|look|looking|find|finding|search|searching|copy|copying|play|pause|mute|muting|skip|"
    r"convert|fetch|fetching|pull|pulling)\b",
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
    "(Note from the system, not sir: do not reuse any line from your examples, and do not "
    "repeat your previous reply. Reply to sir's "
    "message \"{message}\" in your own words, in one or two short sentences.)"
)


def copies_example(sentence: str) -> bool:
    return _words(sentence) in EXAMPLE_LINES


# qwen3 sometimes writes its tool call as text ('{"name": "look_up", "arguments": {...}}',
# often inside a tag or code fence) instead of making it. That must never be spoken;
# it is turned into the real call.
TOOL_TEXT_WRAPPER = re.compile(r"</?[a-z_]+>|```(?:json)?", re.I)
TAGGED_CALL = re.compile(r"^\s*<([a-z_]+)>\s*(\{.*\})\s*</\1>\s*$", re.S)


def tool_call_in_text(text: str):
    """The tool call(s) a reply spells out as JSON, as one ollama Message, else None.
    Only when the reply is nothing but calls to real tools."""
    # '<look_up>{"kind": "definition", "topic": "phishing"}</look_up>': the tag is the tool name
    # and the JSON is just its arguments (seen in the simulation, spoken aloud as-is).
    tagged = TAGGED_CALL.match(text or "")
    if tagged and tagged.group(1) in tools.TOOLS:
        try:
            args = json.loads(tagged.group(2))
        except ValueError:
            args = None
        if isinstance(args, dict) and "name" not in args:
            call = Message.ToolCall(function=Message.ToolCall.Function(name=tagged.group(1), arguments=args))
            return Message(role="assistant", content="", tool_calls=[call])
    body = TOOL_TEXT_WRAPPER.sub(" ", text or "").strip()
    if not body.startswith("{"):
        return None
    decoder, calls, i = json.JSONDecoder(), [], 0
    while i < len(body):
        try:
            data, i = decoder.raw_decode(body, i)
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("name") not in tools.TOOLS:
            return None
        args = data.get("arguments") or data.get("parameters") or {}
        if not isinstance(args, dict):
            return None
        calls.append(Message.ToolCall(function=Message.ToolCall.Function(name=data["name"], arguments=args)))
        while i < len(body) and body[i] in " \n\t\r,":
            i += 1
    return Message(role="assistant", content="", tool_calls=calls) if calls else None


SPEAKER_LABEL = re.compile(r"^(?:You|Jarvis|JARVIS|Assistant)\s*:\s*")
# ---------------------------------------------------------------- context budget
MODEL_OPTIONS = {"num_ctx": config.CONTEXT_TOKENS, "num_predict": config.REPLY_TOKENS}
CHARS_PER_TOKEN = 3.0  # English is ~4; 3 errs on the safe side
CUT_NOTE = " [... cut to fit]"


def _tool_schema_tokens() -> int:
    """Rough size of the tool list the model is sent with every request."""
    return sum(len(fn.__doc__ or "") + len(fn.__name__) + 60 for fn in tools.TOOLS.values()) // 3


def _content(m) -> str:
    if isinstance(m, dict):
        return str(m.get("content") or "") + str(m.get("tool_calls") or "")
    return str(getattr(m, "content", "") or "") + str(getattr(m, "tool_calls", "") or "")


def estimate_tokens(messages: list) -> int:
    return _tool_schema_tokens() + sum(int(len(_content(m)) / CHARS_PER_TOKEN) + 4 for m in messages)


def fit_context(messages: list, history_len: int) -> int:
    """Make the request fit config.CONTEXT_TOKENS so Ollama never cuts the system prompt off.
    Drops the oldest history first, then shortens the longest tool result. The system prompt
    and sir's current message are never touched. Returns how much history is left."""
    budget = config.CONTEXT_TOKENS - config.REPLY_TOKENS
    while estimate_tokens(messages) > budget and history_len > 0:
        del messages[1]
        history_len -= 1
    while estimate_tokens(messages) > budget:
        results = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"
                   and len(m.get("content") or "") > 400]
        if not results:
            break
        longest = max(results, key=lambda m: len(m["content"]))
        over = int((estimate_tokens(messages) - budget) * CHARS_PER_TOKEN) + 50
        keep = max(300, len(longest["content"]) - over)
        longest["content"] = longest["content"][:keep] + CUT_NOTE
    return history_len


# A reply that is nothing but presence filler ("I am here, sir.", "Yes, sir?").
PRESENCE_ONLY = re.compile(
    r"^\s*(?:I(?:'m| am) (?:here|listening)(?: to (?:assist|help|serve)(?: you)?)?|Yes|At your service|How (?:can|may) I (?:help|assist)(?: you)?"
    r"|(?:Is there )?(?:something|anything) (?:I can|you'd like me to) (?:help|assist)(?: you)?(?: with)?)"
    r"(?:,? sir)?[.?!]?\s*$", re.I)
ACKNOWLEDGE = "Very good, sir."
MEMORY_FAILED = "I am afraid that did not save to memory, sir. Would you tell me once more?"
NUDGE = (
    "Sir asked you to remember something, but you did not call the remember_fact tool, "
    "so nothing was saved. Call remember_fact now with the fact, then reply to sir."
)


# A sentence that only offers more help ("If there is something specific you need, please
# let me know.", "I am here to help."). Dropped when it follows a real answer.
SIGN_OFF = re.compile(
    r"\b(?:let me know|feel free|I(?:'m| am) here to (?:help|assist)|I(?:'d| would) be (?:happy|glad) to"
    r"|happy to help|at your disposal|should you need|if you need anything|here to serve)\b", re.I)


def strip_stock_offer(reply: str) -> str:
    """Drop a trailing 'Is there anything else I can assist you with?' (unless that's all he said)."""
    trimmed = TRAILING_OFFER.sub("", STOCK_OFFER.sub("", reply)).strip()
    return trimmed or reply


def warm_up() -> None:
    """Load the model and let it read the system prompt and tool list once, so the first
    real reply doesn't wait for a cold load plus ~3,100 tokens of prompt."""
    chat(model=config.MODEL, messages=[{"role": "system", "content": _system_prompt()},
                                       {"role": "user", "content": "Ready?"}],
         tools=list(tools.TOOLS.values()), think=False, keep_alive=gamemode.keep_alive(),
         options={**MODEL_OPTIONS, "num_predict": 1})


def _system_prompt() -> str:
    """The same text on every request. Facts about sir used to be appended here, but the
    model reads the tool list after the system prompt, so every new fact made Ollama re-read
    ~3,100 tokens of prompt and tools. They now travel with sir's message (facts_note)."""
    return config.SYSTEM_PROMPT


def facts_note(message: str) -> str:
    """The facts about sir that matter for this message, as a note in front of it (not saved
    to the history). Up to config.MEMORY_ALL_FACTS_UP_TO facts all go; beyond that only the
    ones the embedding model finds related (semantic.relevant_facts)."""
    import semantic

    facts = memory.all_facts()
    if not facts:
        return ""
    chosen = semantic.relevant_facts(message, facts)
    return "(Things you know about sir: " + " ".join(f"({i}) {f}" for i, f in enumerate(chosen, 1)) + ")\n\n"


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
    memory.init_db()
    audit_mark = memory.last_audit_id()
    try:
        for event in _respond(message, session, trace):
            if event["type"] == "done":
                # Kept so sir can rate this reply in the HUD (see memory.rate_turn).
                event["turn"] = memory.add_turn(session, message, event["reply"], memory.audit_since(audit_mark))
            yield event
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        memory.add_trace(session, config.MODEL, elapsed, trace["rounds"], trace["tools"],
                         False, f"{type(exc).__name__}: {exc}")
        raise
    elapsed = (time.perf_counter() - started) * 1000
    memory.add_trace(session, config.MODEL, elapsed, trace["rounds"], trace["tools"], True)


HERE_TO = re.compile(r"^I(?:'m|\u2019m| am) here to\b", re.I)
SIR = re.compile(r"\bsir\b", re.I)
EXTRA_SIR = re.compile(r",\s*sir\b(?=\s*[,.?!])", re.I)
LEADING_SIR = re.compile(r"^sir,\s*", re.I)


def _is_filler(text: str) -> bool:
    text = LEADING_SIR.sub("", text.strip())
    return bool(text) and bool(HERE_TO.match(text) or PRESENCE_ONLY.match(text))


class _Sentences:
    """Collects streamed text and releases it a cleaned sentence at a time."""

    def __init__(self):
        self.buffer, self.sent, self.dropped_filler = "", [], False

    def _clean(self, sentence: str) -> str | None:
        sentence = security.redact(sentence.strip())
        if not self.sent:  # "You: A SOC analyst..." copies the examples' "Sir: ... You: ..." layout
            sentence = SPEAKER_LABEL.sub("", sentence)
        if PRESENCE_ONLY.match(sentence):
            self.dropped_filler = True
            return None  # "I am here, sir." says nothing; never speak it
        if HERE_TO.match(LEADING_SIR.sub("", sentence)):
            self.dropped_filler = True
            return None  # "I'm here to help if you need anything" / "...to ensure everything runs smoothly"
        if self.sent and copies_example(sentence):
            return None  # "Ambitious, sir. I'll make no comment on the vegetables." (a copied tail)
        sentence = sentence.replace("!", ".")  # he never exclaims
        if _words(sentence) and any(_words(sentence) == _words(said) for said in self.sent):
            return None  # the model said this sentence already in this reply
        if any(SIR.search(said) for said in self.sent):
            # 'sir' once per reply: "Overcast, sir, and there's a 100%... tomorrow, sir."
            sentence = LEADING_SIR.sub("", EXTRA_SIR.sub("", sentence))
            sentence = sentence[:1].upper() + sentence[1:]
        if (self.sent or self.dropped_filler) and (SIGN_OFF.search(sentence)
                          or not TRAILING_OFFER.sub("", STOCK_OFFER.sub("", sentence)).strip()):
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


# Requests qwen3 kept answering from memory although a tool exists (the same miss in all
# three runs of 'tool_eval --repeat 3'). For these clear cases the request is sent with a
# one-line hint naming the tool. The hint is not saved in the chat history.
TOOL_HINTS = [
    (re.compile(r"\b(clipboard|what did i (just )?cop(y|ied)|what have i (just )?copied)\b", re.I), "clipboard"),
    (re.compile(r"\b(un)?mute\b", re.I), "pc_control"),
    (re.compile(r"\b(what|where) did i (write|note|jot|put down)\b|\b(search|check|look (in|through)|find in) my notes\b"
                r"|\bin my notes\b", re.I), "search_notes"),
    (re.compile(r"^\s*(who (invented|discovered|founded|created|designed|built|wrote|painted)\b(?!.*\byou\b)"
                r"|tell me about (?!yourself|you\b|my\b|your\b))", re.I), "look_up"),
    # "What does ephemeral mean?" got "You may look it up, sir" in two simulation runs.
    (re.compile(r"\bwhat does (?!that|this|it|he|she|they|you|my|your)[\w'-]+ mean\b|^\s*define\b"
                r"|\bmeaning of the word\b", re.I), "look_up"),
]
TOOL_HINT = "\n\n(System hint, not from sir: use the {tool} tool for this rather than answering from memory.)"


# Read-only requests the model ignored even with a hint (0/3 in tool_eval). These are run
# by the brain itself before the model answers, so the model only has to put the result
# into words. Only tools that read and change nothing are routed this way.
_NOTES_TOPIC = re.compile(r"\b(?:what|where) did i (?:write|note|jot|put down)(?: down)? (?:about|on|for|regarding) (.+)"
                          r"|\b(?:search|check|look (?:in|through)) my notes (?:for|about) (.+)", re.I)
_CLIPBOARD_READ = re.compile(r"\bwhat did i (?:just )?cop(?:y|ied)\b|\bwhat(?:'s| is) (?:on|in) (?:my|the) clipboard\b"
                             r"|\bread (?:my|the) clipboard\b", re.I)
_BARE_WEATHER = re.compile(r"^\s*(?:(?:what's|whats|what is|how's|hows|how is) the )?weather(?: like)?"
                           r"(?: today| now| outside| at the moment)?(?:,? jarvis)?\s*\??\s*$", re.I)


def preroute(message: str) -> tuple[str, dict] | None:
    """(tool, args) to run before the model answers, for clear read-only requests."""
    if _CLIPBOARD_READ.search(message):
        return "clipboard", {"action": "read"}
    m = _NOTES_TOPIC.search(message)
    if m:
        topic = re.sub(r"^(?:the|my)\s+", "", (m.group(1) or m.group(2)).strip(" ?.!"), flags=re.I)
        topic = re.sub(r"\s+(?:project|assignment|unit|notes?)$", "", topic, flags=re.I) or topic
        if topic:
            return "search_notes", {"query": topic}
    if _BARE_WEATHER.match(message):
        return "get_weather", {"place": ""}
    return None


def with_tool_hint(message: str) -> str:
    """The message as sent to the model: with a tool hint when one clearly applies."""
    for pattern, tool in TOOL_HINTS:
        if pattern.search(message):
            return message + TOOL_HINT.format(tool=tool)
    return message


def _grounded(name: str, args: dict, messages: list) -> dict:
    """Drop a weather place sir never mentioned. Playing a British butler, the model liked
    to fill in 'London' when sir just said 'Weather?' (seen in the simulation); an empty
    place means his home city."""
    place = str(args.get("place") or "").strip()
    if name == "get_weather" and place:
        said = " ".join(str(m.get("content", "")) for m in messages
                        if isinstance(m, dict) and m.get("role") == "user").lower()
        if place.split(",")[0].strip().lower() not in said:
            args["place"] = ""
    return args


def _instant_command(message: str, session: str, trace: dict) -> str | None:
    """Volume, music, the time...: run it and answer at once, without the big model, when
    the embedding classifier is sure (intents.py). None = not sure, ask the model."""
    if not config.INSTANT_COMMANDS:
        return None
    import intents

    hit = intents.classify(message)
    if hit is None:
        return None
    name = hit[0]
    intent = intents.INTENTS[name]
    trace["tools"].append(intent.tool)
    trace["instant"] = name
    result = tools.run_tool(intent.tool, dict(intent.args), session)
    return intents.reply_for(name, result)


def _respond(message: str, session: str, trace: dict):
    memory.init_db()
    game = gamemode.command(message)
    if game is not None:  # "game mode" / "done gaming": handled in code, no model loaded for it
        reply = gamemode.switch(game)
        memory.add_message(session, "user", security.clean_for_storage("history", message))
        memory.add_message(session, "assistant", reply)
        yield {"type": "sentence", "text": reply}
        yield {"type": "done", "reply": reply}
        return
    settled = _answer_approval(message, session, trace)
    if settled is not None:
        memory.add_message(session, "user", security.clean_for_storage("history", message))
        memory.add_message(session, "assistant", settled)
        yield {"type": "sentence", "text": settled}
        yield {"type": "done", "reply": settled}
        return

    instant = _instant_command(message, session, trace)
    if instant is not None:
        memory.add_message(session, "user", security.clean_for_storage("history", message))
        memory.add_message(session, "assistant", instant)
        yield {"type": "sentence", "text": instant}
        yield {"type": "done", "reply": instant}
        return

    messages = [{"role": "system", "content": _system_prompt()}]
    history = memory.recent_messages(session)
    messages += history
    history_len = len(history)
    messages.append({"role": "user", "content": facts_note(message) + with_tool_hint(message)})
    routed = preroute(message)

    # If sir asked Jarvis to remember something, hold the reply back until the
    # false-memory check below has run, instead of speaking it as it streams.
    hold_back = bool(ASKED_TO_REMEMBER.search(message))
    out = _Sentences()
    reply, called, nudged, pushed_to_act, parrot_retried = "", set(), False, False, False
    cut_off = False
    # His last reply in this conversation. qwen3 repeats it word for word when sir only
    # acknowledges ("ok", "understood", "no problem"): the same error came out four times.
    last_said = next((m["content"] for m in reversed(messages[1:-1]) if isinstance(m, dict)
                      and m.get("role") == "assistant" and m.get("content")), "")
    last_opening = _words(re.split(r"(?<=[.!?])\s+", last_said.strip())[0]) if last_said.strip() else ()

    def is_copy(sentence: str) -> bool:
        repeats_himself = (not called and len(last_opening) >= 4 and _words(sentence) == last_opening)
        return copies_example(sentence) or repeats_himself
    if routed:
        name, args = routed
        called.add(name)
        trace["tools"].append(name)
        yield {"type": "status", "text": STATUS.get(name, "Working on it")}
        call = Message.ToolCall(function=Message.ToolCall.Function(name=name, arguments=args))
        messages.append(Message(role="assistant", content="", tool_calls=[call]))
        routed_result = tools.run_tool(name, args, session)
        messages.append({"role": "tool", "tool_name": name, "content": routed_result})
    for _ in range(config.MAX_TOOL_ROUNDS):
        trace["rounds"] += 1
        text, tool_calls, tool_message, parroted = "", [], None, False
        round_out = _Sentences() if hold_back else out
        history_len = fit_context(messages, history_len)
        for chunk in chat(model=config.MODEL, messages=messages, tools=list(tools.TOOLS.values()),
                          options=MODEL_OPTIONS, keep_alive=gamemode.keep_alive(),
                          think=False, stream=True):
            if getattr(chunk, "done_reason", None) == "length":
                cut_off = True  # hit config.REPLY_TOKENS mid-answer
            piece = chunk.message.content or ""
            text += piece
            if chunk.message.tool_calls:
                tool_calls += list(chunk.message.tool_calls)
                tool_message = chunk.message
            if piece and not tool_calls and not hold_back:
                for sentence in round_out.add(piece):
                    if not parrot_retried and len(round_out.sent) == 1 and is_copy(sentence):
                        parroted = True  # caught before it was spoken
                        break
                    yield {"type": "sentence", "text": sentence}
            if parroted:
                break
        if (not parroted and not parrot_retried and not tool_calls and not hold_back
                and not round_out.sent and is_copy(round_out.buffer)):
            parroted = True  # a one-sentence copy, still unspoken in the buffer
        if (not parroted and not parrot_retried and not tool_calls and not hold_back
                and not round_out.sent and len(_words(message)) >= 3
                and ((round_out.dropped_filler and not round_out.buffer.strip())
                     or _is_filler(round_out.buffer))):
            parroted = True  # he said only "I'm here to help, sir": ask for a real answer once
        if parroted:
            parrot_retried = True
            round_out.dropped_filler = False
            round_out.sent.clear()
            round_out.buffer = ""
            # The retry gets no examples at all: with them still in view, qwen3 often
            # copied the same line a second time (seen in the simulation).
            messages[0] = {"role": "system",
                           "content": messages[0]["content"].replace(config.VOICE_EXAMPLES, "")}
            messages.append({"role": "user", "content": PARROT_NUDGE.format(message=message)})
            continue
        reply = text.strip()
        if not tool_calls:
            spelled = tool_call_in_text(text)
            if spelled is not None:  # nothing of it was spoken: it has no sentence ending
                round_out.buffer = ""
                tool_message, tool_calls, text = spelled, list(spelled.tool_calls), ""
        if tool_calls:
            tool_message.content = text
            tool_message.tool_calls = tool_calls
            messages.append(tool_message)
            for call in tool_calls:
                name = call.function.name
                called.add(name)
                trace["tools"].append(name)
                yield {"type": "status", "text": STATUS.get(name, "Working on it")}
                args = _grounded(name, dict(call.function.arguments or {}), messages)
                if routed and (name, args) == routed:
                    result = routed_result  # already run for this request; don't fetch it twice
                else:
                    result = tools.run_tool(name, args, session)
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
    if cut_off and out.sent:
        out.buffer = ""  # the cap stopped him mid-sentence: don't speak half a sentence
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
