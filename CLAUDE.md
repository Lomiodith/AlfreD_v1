# AlfreD - Voice-Activated AI Assistant

## Project Overview

**AlfreD** is a voice-activated AI assistant in Python. It listens for a wake word, transcribes speech locally with Whisper, routes the request through an intent detector, and answers out loud via streaming TTS. It has web search, file access, web scraping, and persistent memory of past conversations.

User-facing documentation lives in `README.md` — setup, commands, troubleshooting. This file is the internal map.

## Architecture

```
                       alfred_tray.py  (optional system tray wrapper)
                              │
                       AlfreD_v1.py    (entry point, wake/command loop)
                              │
                     conversation_handler.py
                     (routing + conversation state)
                              │
        ┌──────────┬──────────┼──────────┬───────────┐
        │          │          │          │           │
   intent_      audio_     llm_       tts_       tool_manager
   detector    processor  service   service    search_service
                                              memory_manager
                                            performance_monitor
```

`ConversationHandler` is the hub — it owns the conversation history and calls everything else. Services are constructed in `AlfreD_v1.main()` and injected into it.

## Core Components

### Entry points

**`AlfreD_v1.py`** (`main`, lines 14-43) — builds the services, then loops: detect wake word → process command → repeat until a command returns `False` or `stop_event` is set. Cleans up temp audio and shuts down TTS in `finally`.

**`alfred_tray.py`** — pystray wrapper running `main()` on a daemon thread. Start/Stop/Quit menu. Stop sets `AlfreD_v1.stop_event`, which the loop checks each iteration.

### `conversation_handler.py`
The orchestrator. Key methods:

- `process_wake_word_detection` (53-81) — records a short window, transcribes, checks for the wake word
- `_handle_command` (107-140) — the routing table: maps a detected intent to a handler
- `_dispatch_command` (142-180) — the single path to the LLM. Summarizes context if needed, injects retrieved memories and any tool-result system message, streams the response, speaks each sentence, stores the interaction
- `auto_summarize_context` (314-366) — collapses older turns into a summary when the history exceeds `max_context_tokens` (4000)
- `semantic_memory_search` (368-419) — word-overlap scoring over recent memories, with a 60-second cache

**Every tool handler ends by calling `_dispatch_command`** with a `system_message` describing the tool result, so the LLM narrates the outcome instead of the user seeing raw output. `_tool_system_message` builds those consistently.

### `intent_detector.py`
Maps spoken text to an intent **and extracts the argument**. `detect()` (66-94) returns `(intent, query)` in three passes: exact prefix match → substring anywhere → fuzzy `SequenceMatcher` above 0.7 (absorbs speech-to-text errors).

`INTENT_TRIGGERS` is flattened and sorted longest-first once in `__init__`, so specific phrases beat general ones (`"search for"` before `"search"`).

> **Important:** handlers must use the `query` that `detect()` returns. Do not re-slice `command_text` by a hardcoded prefix length — a given intent has many trigger phrases of different lengths, and slicing by one of them corrupts the argument for all the others.

### `audio_processor.py`
Local Whisper (`distil-whisper/distil-large-v3.5`) plus Silero VAD.

- Device auto-selected: CUDA → MPS → CPU, printed at startup
- `record_until_silence` (43-80) — VAD-driven capture; stops after ~2s of silence, hard cap 20s. This is what command recording uses
- `record_audio` — fixed-duration capture, used only for the 1.5s wake-word window
- Model is warmed up at init so the first real transcription isn't slow

### `llm_service.py`
Groq client. Primary `openai/gpt-oss-120b`, fallback `llama-3.3-70b-versatile`.

- `_with_fallback` (23-30) — shared primary-then-fallback wrapper; both entry points go through it
- `_try_streaming` (79-107) — streams deltas, splits on sentence boundaries with the module-level `SENTENCE` regex, invokes `on_sentence` per sentence so TTS starts before the full answer arrives
- `tool_choice="none"` is set deliberately — the system prompt tells the model to answer directly and never emit tool calls

### `tts_service.py`
Edge TTS + pygame playback, per-sentence.

- Language auto-detected per sentence via `langdetect`; `VOICE_MAP` covers 17 languages, falling back to the configured default
- `_clean_for_speech` (117-120) applies the precompiled `SPEECH_SUBSTITUTIONS` table (strips markdown, URLs, table syntax) — **compiled once at module level because it runs on every streamed sentence**
- `_is_speakable` skips symbol-only fragments
- Pressing any key during playback interrupts; the cross-platform keyboard shims sit at the top of the file

### `search_service.py`
DuckDuckGo via `ddgs`. Results cached on disk for 1 hour.

Cache filenames use an **md5 digest, not `hash()`** — the builtin is salted per process, so `hash()`-based keys never survive a restart.

### `tool_manager.py`
System commands, file operations, and scraping — all behind `safe_mode` (on by default).

- `_unsafe_command_reason` (147-161) — word-boundary matching against `DANGEROUS_COMMANDS`, so `rm -rf /` is blocked but `echo confirm` is not
- `web_scraping` (107-145) — **parses the HTML once** and passes the shared `soup` to the extractor
- In `_extract_all_content`, text extraction runs **last**: it calls `decompose()` on `<script>`/`<style>`, mutating the shared tree

### `memory_manager.py`
SQLite for episodic memories, JSON for learned preferences, both under `memory_data/`.

- `_connect` (57-64) — context manager yielding a cursor; commits on success, always closes. **Use it for all database access** rather than hand-rolling connect/commit/close
- `user_preference_learning` (194-206) — runs the four `_analyze_*` passes, then **saves to disk**. The save must stay; without it, preferences only live in memory for the session
- `MEMORY_COLUMNS` keeps INSERT and SELECT column order in sync — don't use `SELECT *`, the row unpacking is positional

### `performance_monitor.py`
Module-level singleton `performance_monitor`. `start_operation` returns an id passed to `end_operation`; thread-safe via a lock, keeps the last 100 metrics, warns above `SLOW_OPERATION_SECONDS`.

### `commands.py`
Static help text rendered by `get_formatted_commands()` for the "show commands" intent.

> **Known drift:** this file advertises commands with no intent trigger, so they are unreachable — `file list` / `file exists` / `file create` / `file delete`, `enable tools` / `disable tools` / `toggle tools`, and `show thinking` / `show thoughts`. `ToolManager` implements list/exists/create/delete, so wiring those up needs only a `file` intent in `INTENT_TRIGGERS` plus a handler branch. The tool-toggle and show-thinking features do not exist at all. `README.md` deliberately documents only what works.

### `config.py`
`SAMPLE_RATE`, `WAKE_WORD`, `WAKE_WORD_DURATION`, `TEMPERATURE` (0.3), `WHISPER_MODEL_ID`, `GROQ_API_KEY`.

Loads `.env.txt` from the project root, or from `sys._MEIPASS` when frozen by PyInstaller. **`GROQ_API_KEY` is the only key required** — search no longer uses Google.

## Key Workflows

**Wake word:** record 1.5s → transcribe → substring match on `WAKE_WORD`.

**Command:** record until silence (max 20s) → transcribe → `IntentDetector.detect()` → handler → `_dispatch_command` → stream response, speaking sentence by sentence → store interaction and learn preferences.

**Tool commands** follow the same path, with the tool's result injected as a system message before the LLM call.

## Data & Storage

```
memory_data/memory.db               episodic memories (SQLite)
memory_data/user_preferences.json   learned preferences
search_cache/<md5>_<n>.json         cached search results (1h TTL)
```

All gitignored, along with `.env.txt`, `__pycache__/`, `assets/`, `.claude/`, `.vscode/`.

## Dependencies

`groq`, `python-dotenv` · `sounddevice`, `scipy`, `numpy<2.0` · `torch`, `transformers`, `accelerate`, `safetensors` · `edge-tts`, `pygame`, `langdetect` · `requests`, `beautifulsoup4`, `ddgs` · `pystray`, `Pillow`

The `torch` in `requirements.txt` is CPU-only on Windows; the CUDA build must be installed separately (see README).

## Conventions

- **Handlers return `bool`** — `True` to keep listening, `False` to shut down. Only `terminate` returns `False`.
- **Tool methods return a dict** with `success` plus either result fields or `error`. Callers check `result['success']` before reading anything else.
- **Failures are caught and reported, not raised** — a failed command should never kill the listening loop.
- **Regexes that run per sentence or per chunk are compiled at module level**, not inside the function.
- Comments are sparse by design: explain *why* (the ordering constraint in `_extract_all_content`, the md5 rationale), not *what*.

## Testing

There is no test suite. When changing routing, parsing, or extraction logic, verify against the real behaviour rather than by inspection — stub the services with `MagicMock` and assert on what each tool method was called with. `IntentDetector` in particular is worth snapshotting before and after a change, since its three-pass matching is easy to alter accidentally.

## Possible Enhancements

- Wire up a `file` intent so the documented file-management commands work
- Lightweight local wake-word model (currently full Whisper on every 1.5s window)
- Continuous conversation mode — skip re-triggering the wake word for follow-ups
- Vector embeddings for semantic memory instead of word overlap
- Automatic `cleanup_old_memories()` scheduling (implemented, never called)
