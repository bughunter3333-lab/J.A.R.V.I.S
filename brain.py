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


def respond(message: str, session: str = "default") -> str:
    """Take what the user said, return Jarvis's reply, and save both to memory.

    Every request is traced (latency, model rounds, tools used) so slow or failing
    turns can be spotted later with GET /stats.
    """
    started = time.perf_counter()
    trace = {"rounds": 0, "tools": []}
    try:
        reply = _respond(message, session, trace)
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        memory.add_trace(session, config.MODEL, elapsed, trace["rounds"], trace["tools"],
                         False, f"{type(exc).__name__}: {exc}")
        raise
    elapsed = (time.perf_counter() - started) * 1000
    memory.add_trace(session, config.MODEL, elapsed, trace["rounds"], trace["tools"], True)
    return reply


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


def _respond(message: str, session: str, trace: dict) -> str:
    memory.init_db()
    settled = _answer_approval(message, session, trace)
    if settled is not None:
        memory.add_message(session, "user", security.clean_for_storage("history", message))
        memory.add_message(session, "assistant", settled)
        return settled
    messages = [{"role": "system", "content": _system_prompt()}]
    messages += memory.recent_messages(session)
    messages.append({"role": "user", "content": message})

    reply, called, nudged = "", set(), False
    for _ in range(config.MAX_TOOL_ROUNDS):
        trace["rounds"] += 1
        response = chat(
            model=config.MODEL,
            messages=messages,
            tools=list(tools.TOOLS.values()),
            think=False,
        )
        reply = (response.message.content or "").strip()
        if response.message.tool_calls:
            messages.append(response.message)
            for call in response.message.tool_calls:
                name = call.function.name
                called.add(name)
                trace["tools"].append(name)
                result = tools.run_tool(name, dict(call.function.arguments or {}), session)
                messages.append({"role": "tool", "tool_name": name, "content": result})
            continue
        false_claim = (
            ASKED_TO_REMEMBER.search(message)
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

    # Never echo or store a secret, even if the model repeats one back.
    reply = security.redact(strip_stock_offer(reply))
    memory.add_message(session, "user", security.clean_for_storage("history", message))
    memory.add_message(session, "assistant", reply)
    return reply
