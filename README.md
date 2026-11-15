# J.A.R.V.I.S.

JARVIS-X is a local AI assistant inspired by Tony Stark's J.A.R.V.I.S. It combines conversational AI, voice control, memory, tools and workflow automation, and it runs on your own PC with free, local models.

## What works now

- **HUD interface** (`ui/index.html`): a blue holographic core that reacts while Jarvis listens, thinks and speaks.
- **Brain:** Ollama running `qwen3:8b`, with streamed replies.
- **Voice out:** Kokoro (`bm_george` British voice) through Kokoro-FastAPI. If it isn't running, the browser's built-in voice takes over.
- **Voice in:** the browser's speech recognition (Chrome). This will be replaced by local RealtimeSTT.

## Run it (Windows)

1. Make sure Ollama is running and the model is installed:
   ```
   ollama pull qwen3:8b
   ```
2. (Optional, for the proper voice) start Kokoro-FastAPI on the GPU:
   ```
   docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
   ```
3. Serve the interface from the `ui` folder:
   ```
   cd ui; python -m http.server 8000
   ```
4. Open http://localhost:8000 in Chrome. Hold **Space** to talk, **Enter** to send typed text, **Esc** to stop.

The page must be opened from `localhost`, not by double-clicking the file, or the microphone and Ollama calls won't work.

## Layout

| Folder | Purpose |
|---|---|
| `ui/` | HUD front end |
| `core/` | Brain: Ollama chat loop, tools, memory, audit log |
| `voice/` | Speech in and out services |

## Roadmap

1. Chat, personality and voice (in progress)
2. Wake word ("hey jarvis") and SQLite memory
3. Three safe tools with an allow-list, approval step and audit log
4. Multi-step tasks (LangGraph), screen understanding, browser control
5. Home Assistant and monitoring

## Security

Rules are enforced in code, not just in the prompt: allow-listed apps, folders and commands; human approval before anything destructive; never running raw model output; an audit log of every tool call; and everything bound to `127.0.0.1`.
