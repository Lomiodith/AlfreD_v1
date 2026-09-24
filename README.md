# AlfreD

A voice assistant for your desktop. Say **"Alfred"**, ask for something in plain English or Romanian, and it answers out loud: searching the web, reading your email and calendar, checking GitHub, setting timers, running commands and remembering what you tell it.

Listening runs locally: speech recognition (Parakeet or Whisper), voice-activity detection and end-of-turn detection. The language model runs locally through llama.cpp by default, with Groq models as fallbacks (or the other way round; see `LLM_CHAIN`). Speech is Kokoro (local) for English and Microsoft Edge voices for other languages.

---

## Requirements

- **Python 3.12** with an **NVIDIA GPU** recommended (runs on CPU, slowly)
- A **microphone**
- A **Groq API key**: free at [console.groq.com](https://console.groq.com)
- *Optional:* [Git for Windows](https://git-scm.com/) (bash commands), the [GitHub CLI](https://cli.github.com/) (GitHub tools), a Google Cloud OAuth client (Gmail and Calendar), a [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) CUDA build (local model)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pulls the **CPU** build of PyTorch on Windows. With an NVIDIA GPU, install the CUDA build instead:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

AlfreD picks the device automatically and prints it at startup (`🔧 Using device: cuda`).

### 2. Configure

Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`. Every other setting (models, voices, timings, storage folders) lives in `.env` too; see [Configuration](#configuration).

`.env` is gitignored. Never commit it.

### 3. Optional integrations

**Gmail and Google Calendar.** In Google Cloud Console, enable the Gmail and Calendar APIs, create an OAuth client of type *Desktop app*, and save its JSON as `google_credentials.json` in `SECRETS_DIR`. Then sign in once:

```bash
python google_tools.py
```

Gmail access is read-only; Calendar can list and add events. While your Google app is in *Testing* mode, Google expires the sign-in after 7 days; run the command again, or publish the app.

**GitHub.** Sign in to the GitHub CLI once (`gh auth login`). AlfreD uses that login, read-only.

**Local model (default; Groq models are the fallback).** Download a [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) CUDA build and a GGUF model (currently `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q5_K_M.gguf` from `empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF`). Set `LLAMACPP_SERVER` to `llama-server.exe`, `LLAMACPP_MODEL` to the `.gguf` file, and list `llamacpp:<any name>` in `LLM_CHAIN`. AlfreD starts the server at startup, keeps the model in GPU memory, and stops it on exit.

### 4. First run

The first launch downloads the speech, voice-activity, end-of-turn, voice and embedding models (a few GB, into `MODELS_DIR`). Later starts take about 30 seconds.

---

## Running

```bash
python alfred_tray.py      # system tray icon with Start / Stop / Quit
python AlfreD_v2.py        # console; Ctrl+C or say "exit" to quit
```

---

## Talking to AlfreD

1. Say **"Alfred"**. It listens in 1.5-second windows and ignores silence.
2. After `✅ Wake word detected!`, just talk. It notices when you've finished a sentence (Smart Turn) and waits if you pause mid-thought.

```
Heard: "Alfred."
✅ Wake word detected! Listening for command...
Command heard: "Set a timer for ten minutes for the pasta."
🔥 Alfred: Done, a ten-minute pasta timer is running.
```

There are no fixed phrases: ask naturally and the model picks the right tool. Some things to try:

| Ask | What happens |
|---|---|
| "What's the weather in Cluj this weekend?" | Web search |
| "Do I have any unread emails?" · "What's on my calendar tomorrow?" | Gmail / Calendar |
| "Add dentist on Friday at three" | Creates a calendar event |
| "Any GitHub notifications?" | GitHub |
| "Set a timer for five minutes" · "Cancel the pasta timer" | Timers, announced when they finish |
| "Remember that I'm 33" · "What do you know about me?" | Long-term memory |
| "Run git status in bash" · "Read config.py" | Commands and files (safe mode) |
| "Open alfred tray" · "Open the budget spreadsheet" | Opens a file or folder in its default program; asks which one if several match |
| "Switch to text mode" | Full answers on screen instead of short spoken ones |

**Short spoken answers.** In voice mode Alfred answers in two or three sentences and asks *"Want more?"* when there is more. Say "yes" or "no" without saying "Alfred" again. In **text mode** the full answer is printed and nothing is spoken.

**Other languages.** Speak English or Romanian (`ALLOWED_LANGUAGES`) and Alfred replies in the same language. If a transcript comes out in another language, Alfred asks whether you meant to speak it.

**Interrupting:** press any key while Alfred is answering. It stops talking, cancels anything it hadn't started yet, and listens for your next request without the wake word.

### Instant commands

These are handled without the language model, and only when they are the whole sentence:

| Say | Does |
|---|---|
| `exit` · `quit` · `goodbye` · `shut down` | Shuts AlfreD down |
| `clear context` · `start fresh` · `new conversation` | Forgets the current conversation (long-term memory stays) |
| `text mode` · `voice mode` | Switches output mode |
| `show commands` · `what can you do` | Prints the command reference |
| `performance stats` | Prints timing statistics |
| `no` · `nope` · `that's all` (after a question) | Ends the exchange |

---

## Memory

Stored locally in `memory_data/`:

- **Conversation memory:** your past questions, searchable by meaning in any language, so related topics are brought back automatically. Alfred's past answers are never reused, so a wrong answer can't repeat itself.
- **Saved facts:** only things you explicitly ask it to remember. "Forget that…" deletes one.
- **Preferences:** patterns it notices, such as brief vs. detailed answers.

Long conversations are summarised automatically. Delete `memory_data/` to erase everything.

---

## Configuration

All settings are in `.env` (see `.env.example` for the full list with comments). The main ones:

| Setting | Default | What it does |
|---|---|---|
| `STT_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | Speech recognition. `distil-whisper/distil-large-v3.5` is English-only |
| `ALLOWED_LANGUAGES` | `en,ro` | Languages you speak |
| `SMART_TURN` | `true` | End your turn when you sound finished, not after a fixed silence |
| `SILENCE_SECONDS` | `2.0` | Fallback end-of-turn silence |
| `TTS_ENGINE` / `KOKORO_VOICE` / `KOKORO_DEVICE` | `kokoro` / `bm_george` / `cuda` | Local voice for English; other languages use Edge voices |
| `TTS_VOICE` | `en-GB-RyanNeural` | Edge voice used when Kokoro isn't |
| `VOICE_OUTPUT` | `true` | Start in voice mode (`false` = text mode) |
| `LLM_CHAIN` | `llamacpp:qwythos-9b,groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b` | Models tried in order; the next is used when one is rate-limited or fails |
| `LLAMACPP_SERVER` / `LLAMACPP_MODEL` / `LLAMACPP_ARGS` | — | llama.cpp server AlfreD starts for a `llamacpp:` model |
| `LLM_REASONING_VOICE` / `LLM_REASONING_TEXT` | `low` / `medium` | Thinking before answering, per mode. Qwen can only think or not: below `medium` means off |
| `MODELS_DIR` / `SECRETS_DIR` | — | Where downloaded models and the Google sign-in are kept |
| `FILE_SEARCH_DIRS` | Desktop; Documents; Downloads; AlfreD folder | Folders searched for "open <file>", separated by `;` |

---

## Safety

Commands and file operations run in **safe mode**: destructive commands (`rm`, `del`, `format`, `shutdown`, …) are blocked, Unix system directories are off limits, and commands time out after 30 seconds. Windows paths are not restricted. Scraping allows only `http`/`https` and blocks localhost. Gmail is read-only, and calendar events are only created when you ask for one.

---

## Troubleshooting

**It never hears the wake word.** Check the input device. `Heard: "..."` shows each transcription; silence is skipped entirely, so no line means it heard nothing.

**It cuts me off, or waits too long.** Raise or lower `SMART_TURN_THRESHOLD` (0.5), or set `SMART_TURN=false` to use the fixed `SILENCE_SECONDS`.

**"I can't reach the language model right now."** Every model in the chain was rate-limited or unreachable. Groq's free tier has tight per-minute limits, and a local model in `LLM_CHAIN` removes the problem. If the local model fails, check `logs/alfred.log` for llama-server errors.

**It mishears place names.** Tell it once ("Remember that I live in Cluj-Napoca") and the model will resolve close-sounding transcriptions.

Logs, including each spoken answer, are in `logs/alfred.log`.

---

## Project layout

```
AlfreD_v2.py            Entry point and main loop
alfred_tray.py          System tray wrapper
config.py               Settings, loaded from .env

audio_processor.py      Recording, voice activity (Silero), speech recognition
smart_turn.py           End-of-turn detection
conversation_handler.py Conversation flow, voice/text modes, follow-ups
intent_detector.py      Instant control commands
language_guard.py       Flags transcripts in unexpected languages
llm_service.py          Model chain, streaming, tool-calling loop
tts_service.py          Speech output (Kokoro / Edge), pipelined

tools.py                Tool registry + core tools (search, commands, files, timers, memory)
file_opener.py          Finds a file by its spoken name and opens it
github_tools.py         GitHub tools (gh CLI)
google_tools.py         Gmail and Calendar tools
timers.py               Countdown timers
search_service.py       Web search with caching
tool_manager.py         Commands, files, scraping (safe mode)
memory_manager.py       Memory, saved facts, preferences
embeddings.py           Meaning vectors for memory search
performance_monitor.py  Operation timing
commands.py             Command reference text
utils.py                Logging setup, stop signal, temp cleanup
```
