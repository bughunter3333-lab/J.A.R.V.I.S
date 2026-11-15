# J.A.R.V.I.S

JARVIS-X is an AI-powered personal operating system inspired by Tony Stark's J.A.R.V.I.S. It combines conversational AI, voice control, memory, autonomous agents, and workflow automation to help users code, manage projects, execute tasks, and interact with applications through natural language.

The language model runs on this PC through Ollama, so your conversations stay on this machine. Only the internet tools reach out: search queries go to DuckDuckGo, weather lookups go to Open-Meteo, and pages you ask about are fetched directly. Set `ALLOW_INTERNET = False` in `config.py` to switch all of that off.

## What he can do

| Ask something like | Tool |
|---|---|
| "What's the time?" | `get_current_time` |
| "Open notepad" | `open_app` (allow-list only) |
| "Make a note called shopping…", "Find my note about…" | `create_note`, `search_notes` |
| "Remember that my exam is on Friday" | `remember_fact` |
| "Search for the latest on…", "Any news about…" | `web_search` (DuckDuckGo, no key needed) |
| "Read this page: https://…" | `read_webpage` |
| "What's the weather?", "Will it rain in Melbourne tomorrow?" | `get_weather` (Open-Meteo, no key needed) |
| "What's 15% of 1,250?" | `calculate` (safe, never runs code) |
| "How's the computer doing?" | `system_status` (CPU, memory, disk, battery, GPU) |
| "Remind me at 3pm to submit the report", "Set a 10 minute timer", "What reminders do I have?" | `reminder` (Windows notification + he says it in the HUD; survives restarts) |
| "Turn it up", "Pause the music", "Lock the screen", "Open YouTube" | `pc_control` (fixed actions only, no shutdown or delete) |
| "100 AUD in taka?", "5 km in miles", "30°C in Fahrenheit" | `convert` (live daily exchange rates, no key needed) |
| "Who was Alan Turing?", "Define serendipity" | `look_up` (Wikipedia, dictionary) |
| "Find my assessment 3 file", "Read and summarise it" | `files` (only inside `ALLOWED_FOLDERS`; txt, md, csv, pdf, docx…) |
| "What did I just copy?", "Copy that to my clipboard" | `clipboard` |
| "How long ago were you built?", "What version are you?" | `about_me` (exact to the second, worked out in code; set `BUILT_AT` in `config.py` to pin the exact moment) |

## Run it

```powershell
cd D:\jarvis-x; .\.venv\Scripts\Activate.ps1; uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs and use **POST /chat**. **GET /reminders** lists waiting reminders. **GET /audit** shows every tool call Jarvis made or refused, and **GET /stats** shows response times and which tools he used.

## Something not working?

```powershell
cd D:\jarvis-x; .\.venv\Scripts\Activate.ps1; python doctor.py
```

It checks Ollama, the model, the model folder, the GPU, packages, the voice model and the data folder, and prints the exact fix for anything wrong. The HUD runs the same checks through **GET /health** when it loads.

## The HUD

With the server running, open http://127.0.0.1:8000/hud (any browser). It shows a blue holographic core that reacts while Jarvis listens, thinks and speaks.

- **Talking:** click the mic (or press Space) once to switch on listening; it stays on. Say **"Jarvis"** and your request, then just keep talking: follow-ups need no wake word. Say **"that's all, thank you"** (or "that will be all", "goodbye Jarvis", "we're done") to end the conversation; after that he waits for "Jarvis" again. The wake word is `WAKE_PHRASE` in `config.py`: `"jarvis"` (Porcupine's built-in keyword, offline, no account) or `"hey jarvis"` (openWakeWord); `WAKE_SENSITIVITY` (0–1) trades missed wake-ups against false ones. If nobody speaks for 2 minutes the conversation ends by itself, so a TV or someone else in the room can't keep giving commands (`CONVERSATION_IDLE_TIMEOUT` in `config.py`, 0 = never). The server listens on this PC's microphone with RealtimeSTT, the same as `talk.py`, so nothing goes to Google and it works in Firefox. The mic is muted while he speaks. In ⚙ settings you can turn off the wake word, or switch back to the browser's own speech recognition (Chrome/Edge, which sends audio to Google).
- **Typing:** press Enter to send, and Esc to stop him.
- **Replies stream:** the status line shows what he's doing ("Searching the web…"), and he starts speaking after the first sentence.
- **Approvals:** for anything risky he asks first and shows **Approve / Deny**. Saying "yes" or "no" works too.
- Replies go through `/chat/stream`, so tools, memory, approvals and the audit log all apply. His voice comes from `/speak` (Kokoro), with the browser voice as a fallback.

## Talk to him

```powershell
cd D:\jarvis-x; .\.venv\Scripts\Activate.ps1; python talk.py
```

Say "Jarvis", then your request. Say "goodbye Jarvis" or press Ctrl+C to stop.
`python talk.py --no-wake` listens without the wake word. `python talk.py --text` lets you type while he answers out loud.

## Files

| File | Job |
|---|---|
| `config.py` | All settings: model, personality, allowed apps, voice |
| `main.py` | Web API (`/chat`, `/chat/stream`, `/speak`, `/listen/*`, `/reminders`, `/audit`, `/stats`, `/health`, `/hud`) |
| `brain.py` | Thinking loop: history + facts + model + tools |
| `memory.py` | SQLite storage: messages, facts, audit log |
| `tools.py` | Actions Jarvis can take, with the security rules |
| `web.py` | Internet tools: search, page reading, weather, Wikipedia, dictionary, exchange rates |
| `pc.py` | PC control, clipboard, finding and reading files, with their rules |
| `reminders.py` | Reminders: time parsing, storage, Windows notifications |
| `security.py` | Secret, personal-ID and prompt-injection scanners |
| `doctor.py` | Health check with fixes |
| `voice.py` | Text to speech (Kokoro, British voice) |
| `talk.py` | Voice conversation in the terminal: wake word, speech recognition, spoken replies |
| `listener.py` | The same listening, run by the server for the HUD |
| `approvals.py` | Asking sir before risky actions |
| `evals/tool_eval.py` | Tool-picking test against the real model |
| `ui/index.html` | The HUD: holographic interface served at `/hud` |
| `data/` | Database, notes workspace, audio (not committed to Git) |

## Security rules (enforced in code)

- Apps: only those listed in `config.ALLOWED_APPS` can be opened.
- Files: notes are written only inside `data/workspace`, as `.txt`, and never overwrite.
- Every tool call, allowed or refused, is written to the audit log, with secrets redacted.
- Secrets and ID numbers (API keys, tokens, passwords, card, TFN and Medicare numbers) are never stored in memory, notes or history, and are redacted from replies. Each catch is logged as a `security` audit entry, without the secret itself.
- Note contents are handed to the model as untrusted data, and text that looks like a prompt injection ("ignore previous instructions", fake `system:` lines, exfiltration requests) is flagged to the model and logged.
- The model cannot claim it remembered something unless the tool really ran.
- Search queries and links are scanned before they leave the PC; one containing a password, key or ID number is refused, not sent.
- `read_webpage` only opens public http(s) pages on ports 80/443. Every address a link resolves to, including after redirects, is checked, and this PC, the home network, link-local and cloud-metadata addresses are refused (server-side request forgery protection). Pages are size- and time-capped, and only text/HTML is read.
- Search results and pages reach the model inside `<untrusted>` tags and are scanned for prompt injection first.
- `calculate` parses the maths itself and never calls `eval`, so it can't be used to run code.
- `pc_control` can only do its fixed list of actions; there is deliberately no shutdown, restart, delete or "run a command".
- `open_website` refuses `javascript:`, `file:`, `data:` and other non-web links, links with passwords in them, and addresses on this PC or the home network.
- `files` only works inside `ALLOWED_FOLDERS` (Documents, Desktop, Downloads by default) and the notes workspace. Paths are fully resolved first, so `..` and symlinks can't escape. Secret-looking files (`.env`, `.key`, `.pem`, `.kdbx`, SSH keys, anything named "password") are hidden from search and refused.
- Text from files and the clipboard is scanned, secrets are redacted, and it reaches the model as untrusted data. Clipboard text is written through stdin, never pasted into a command.
- The server listens on 127.0.0.1 only.

## Asking before acting

Some actions wait for sir's permission (`NEEDS_APPROVAL` in `config.py`): locking the screen, opening a website, writing to the clipboard and saving notes. The tool doesn't run. Jarvis asks, and only sir's next message decides: "yes" runs it, "no" cancels it, and anything else (or 2 minutes) drops it. That decision is made by code reading sir's words, never by the model, so a web page that tells him to "approve" something can't.

## How good is he at picking tools?

```powershell
cd D:\jarvis-x; .\.venv\Scripts\Activate.ps1; python evals\tool_eval.py
```

This asks the real model 41 typical requests and checks which tool it picks first (nothing is actually run). It prints a score, per-tool results and the most common mistakes, and saves the full results in `data\evals\`. Run it again after changing tools or the prompt, to see whether things got better or worse.

## Tests

```powershell
python -m pytest -q            # offline, no model needed
python tests\smoke_live.py     # live check against the real model
python tests\voice_roundtrip.py  # text -> speech -> text, no mic or speakers needed
python tests\recorder_check.py   # wake word + listening pipeline, fed from a recording
```

## Credits

`security.py` (scanner patterns) and `doctor.py` (the idea of a health check with fixes) are adapted from [OpenJarvis](https://github.com/open-jarvis/OpenJarvis) by Stanford's Hazy Research and Scaling Intelligence Lab, Apache-2.0 licensed.
