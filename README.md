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

## Run it

```powershell
cd D:\jarvis-x; .\.venv\Scripts\Activate.ps1; uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs and use **POST /chat**. **GET /audit** shows every tool call Jarvis made or refused, and **GET /stats** shows response times and which tools he used.

## Something not working?

```powershell
cd D:\jarvis-x; .\.venv\Scripts\Activate.ps1; python doctor.py
```

It checks Ollama, the model, the model folder, the GPU, packages, the voice model and the data folder, and prints the exact fix for anything wrong. The HUD runs the same checks through **GET /health** when it loads.

## The HUD

With the server running, open http://127.0.0.1:8000/hud in Chrome. It shows a blue holographic core that reacts while Jarvis listens, thinks and speaks.
Hold **Space** (or click the mic) to talk, press **Enter** to send typed text, and **Esc** to stop him. **AUTO** keeps listening after each reply.
Replies go through `/chat`, so tools, memory and the audit log all apply. His voice comes from `/speak` (Kokoro); if that fails, the browser's own voice takes over.
Speech input uses Chrome's built-in recognition, which sends audio to Google. For fully local listening, use `talk.py`.

## Talk to him

```powershell
cd D:\jarvis-x; .\.venv\Scripts\Activate.ps1; python talk.py
```

Say "Hey Jarvis", then your request. Say "goodbye Jarvis" or press Ctrl+C to stop.
`python talk.py --no-wake` listens without the wake word. `python talk.py --text` lets you type while he answers out loud.

## Files

| File | Job |
|---|---|
| `config.py` | All settings: model, personality, allowed apps, voice |
| `main.py` | Web API (`/chat`, `/speak`, `/audit`, `/stats`, `/health`, `/hud`) |
| `brain.py` | Thinking loop: history + facts + model + tools |
| `memory.py` | SQLite storage: messages, facts, audit log |
| `tools.py` | Actions Jarvis can take, with the security rules |
| `web.py` | Internet tools: search, page reading, weather, with the network rules |
| `security.py` | Secret, personal-ID and prompt-injection scanners |
| `doctor.py` | Health check with fixes |
| `voice.py` | Text to speech (Kokoro, British voice) |
| `talk.py` | Voice conversation: wake word, speech recognition, spoken replies |
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
- The server listens on 127.0.0.1 only.

## Tests

```powershell
python -m pytest -q            # offline, no model needed
python tests\smoke_live.py     # live check against the real model
python tests\voice_roundtrip.py  # text -> speech -> text, no mic or speakers needed
python tests\recorder_check.py   # wake word + listening pipeline, fed from a recording
```

## Credits

`security.py` (scanner patterns) and `doctor.py` (the idea of a health check with fixes) are adapted from [OpenJarvis](https://github.com/open-jarvis/OpenJarvis) by Stanford's Hazy Research and Scaling Intelligence Lab, Apache-2.0 licensed.
