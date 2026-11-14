# JARVIS-X

A local, private voice assistant. The language model runs on this PC through Ollama, so nothing leaves the machine.

## Run it

```powershell
cd D:\jarvis-x; .\.venv\Scripts\Activate.ps1; uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs and use **POST /chat**. **GET /audit** shows every tool call Jarvis made or refused.

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
| `main.py` | Web API (`/chat`, `/audit`) |
| `brain.py` | Thinking loop: history + facts + model + tools |
| `memory.py` | SQLite storage: messages, facts, audit log |
| `tools.py` | Actions Jarvis can take, with the security rules |
| `voice.py` | Text to speech (Kokoro, British voice) |
| `talk.py` | Voice conversation: wake word, speech recognition, spoken replies |
| `data/` | Database, notes workspace, audio (not committed to Git) |

## Security rules (enforced in code)

- Apps: only those listed in `config.ALLOWED_APPS` can be opened.
- Files: notes are written only inside `data/workspace`, as `.txt`, and never overwrite.
- Every tool call, allowed or refused, is written to the audit log.
- The model cannot claim it remembered something unless the tool really ran.
- The server listens on 127.0.0.1 only.

## Tests

```powershell
python -m pytest -q            # offline, no model needed
python tests\smoke_live.py     # live check against the real model
python tests\voice_roundtrip.py  # text -> speech -> text, no mic or speakers needed
python tests\recorder_check.py   # wake word + listening pipeline, fed from a recording
```
