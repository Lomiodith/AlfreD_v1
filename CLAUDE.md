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

**`AlfreD_v1.py`** (`main`) — builds the services, then loops: detect wake word → process command → repeat until a command returns `False` or `stop_event` is set. Cleans up temp audio and shuts down TTS in `finally`.

**`alfred_tray.py`** — pystray wrapper running `main()` on a daemon thread. Start/Stop/Quit menu. Whether Alfred is running is read from the thread itself (`_is_running`), not a flag, so Start can't launch a second instance while the first is still winding down.

**Stopping** — `utils.stop_event` is the shared stop signal (it lives in `utils` so services can import it without a cycle through `AlfreD_v1`). Every blocking stage polls it: both recorders, the LLM stream (which also skips the fallback model once stopped), TTS synthesis and playback, and the handler between recording and transcription. Only an in-progress Whisper transcription or the startup model load can't be cut short. Anything new that blocks for more than a moment should check it too.

### `conversation_handler.py`
The orchestrator. Key methods:

- `process_wake_word_detection` — records a short window, transcribes, checks for the wake word
- `_handle_command` — the routing table: maps a detected intent to a handler
- `_dispatch_command` — the single path to the LLM. Summarizes context if needed, builds the request (history + retrieved memories + any tool-result system message + the user turn), streams the response, speaks each sentence, stores the interaction
- `auto_summarize_context` — collapses older turns into a summary when the history exceeds `MAX_CONTEXT_TOKENS` (4000), keeping the last `RECENT_MESSAGES_KEPT` (10)
- `semantic_memory_search` — word-overlap scoring over recent memories, with a 60-second cache

**Every tool handler ends by calling `_dispatch_command`** with a `system_message` describing the tool result, so the LLM narrates the outcome instead of the user seeing raw output. `_tool_system_message` builds those consistently.

**Tool results and retrieved memories are sent with their one request only, never stored in `conversation_history`.** The history holds just the system prompt, any summaries, and user/assistant turns; the assistant's answer is what carries a tool result forward. Persisting them made every later request resend every earlier search result, and summarization couldn't shed them because it keeps all system messages.

### `intent_detector.py`
Maps spoken text to an intent **and extracts the argument**. `detect()` returns `(intent, query)` in three passes: prefix match → match anywhere in the text → fuzzy `SequenceMatcher` on the leading words above 0.9 (absorbs speech-to-text errors). Only the prefix pass extracts an argument, and it trims `QUERY_EDGE_PUNCTUATION` from both ends, since Whisper punctuates transcripts (`"Read file notes.txt."` → `notes.txt`).

`INTENT_TRIGGERS` is flattened and sorted longest-first once in `__init__`, so specific phrases beat general ones (`"search for"` before `"search"`). Each trigger is also compiled there into a `\b…\b` regex, so the first two passes match **whole words only** — `"quit"` no longer fires on `"quitter"`, `"exit"` not on `"exits"`. The fuzzy pass threshold is `FUZZY_THRESHOLD = 0.9` (raised from 0.7, which let unrelated first words route to intents).

> **Important:** handlers must use the `query` that `detect()` returns. Do not re-slice `command_text` by a hardcoded prefix length — a given intent has many trigger phrases of different lengths, and slicing by one of them corrupts the argument for all the others.

### `audio_processor.py`
Local Whisper (`distil-whisper/distil-large-v3.5`) plus Silero VAD.

- Device auto-selected: CUDA → MPS → CPU, printed at startup. The model loads in float32 on every device
- `record_until_silence` — VAD-driven capture; stops after ~2s of silence, hard cap 20s. This is what command recording uses
- `record_audio` — fixed-duration capture, used only for the 1.5s wake-word window
- Model is warmed up at init so the first real transcription isn't slow

### `llm_service.py`
Groq client. Primary `openai/gpt-oss-120b`, fallback `llama-3.3-70b-versatile`.

- `_with_fallback` — shared primary-then-fallback wrapper; both entry points go through it
- `_try_streaming` — streams deltas, splits on sentence boundaries with the module-level `SENTENCE` regex, invokes `on_sentence` per sentence so TTS starts before the full answer arrives
- `tool_choice="none"` is set deliberately — the system prompt tells the model to answer directly and never emit tool calls

### `tts_service.py`
Edge TTS + pygame playback, per-sentence.

- Language auto-detected per sentence via `langdetect`; `VOICE_MAP` covers 17 languages, falling back to the configured default
- `_clean_for_speech` applies the precompiled `SPEECH_SUBSTITUTIONS` table (strips markdown, URLs, table syntax) — **compiled once at module level because it runs on every streamed sentence**
- `_is_speakable` skips symbol-only fragments
- Pressing any key during playback interrupts; the cross-platform keyboard shims sit at the top of the file

### `search_service.py`
DuckDuckGo via `ddgs`. The public surface is `search_ranked(query, n)` → a relevance-sorted list of `{title, link, snippet}`, and `format_results(items, query)` → the system message for the LLM. Results are cached on disk for `CACHE_SECONDS` (1 hour); cache files without an `items` key (the older format) are treated as misses.

Cache filenames use an **md5 digest, not `hash()`** — the builtin is salted per process, so `hash()`-based keys never survive a restart.

### `tool_manager.py`
System commands, file operations, and scraping — all behind `safe_mode` (on by default).

- `execute_system_command` runs the command string through **bash** (`bash -c`), so pipes, `&&` and Unix tools work. `BASH` is resolved once at import by `_find_bash`: on Windows it walks up from `git` to Git for Windows' `bin/bash.exe`, because a plain PATH lookup there finds WSL's `System32/bash.exe` first. Elsewhere it's the system `bash`. Without bash the method returns an error dict
- Safety is a text check on the whole command string, so chaining can't smuggle a blocked word past it. `_is_path_safe` only knows Unix paths; Windows paths (`C:\...`) all pass

- `_unsafe_command_reason` — word-boundary matching against `DANGEROUS_COMMANDS`, so `rm -rf /` is blocked but `echo confirm` is not
- `web_scraping` — **parses the HTML once** and passes the shared `soup` to the extractor
- In `_extract_all_content`, text extraction runs **last**: it calls `decompose()` on `<script>`/`<style>`, mutating the shared tree

### `memory_manager.py`
SQLite for episodic memories, JSON for learned preferences, both under `memory_data/`.

- `_connect` — context manager yielding a cursor; commits on success, always closes. **Use it for all database access** rather than hand-rolling connect/commit/close
- `user_preference_learning` — runs the four `_analyze_*` passes, then **saves to disk**. The save must stay; without it, preferences only live in memory for the session
- `MEMORY_COLUMNS` keeps INSERT and SELECT column order in sync — don't use `SELECT *`, the row unpacking is positional. This also keeps older `memory.db` files working: they still carry the since-removed `embedding` column, which the explicit column list simply ignores
- Older databases still contain an unused `memory_metadata` table; it is no longer created or read

### `performance_monitor.py`
Module-level singleton `performance_monitor`. `start_operation` returns an id passed to `end_operation`; thread-safe via a lock, keeps the last 100 metrics, warns above `SLOW_OPERATION_SECONDS`.

### `commands.py`
Static help text rendered by `get_formatted_commands()` for the "show commands" intent.

Keep it in sync with `INTENT_TRIGGERS`: it lists only commands that are actually reachable, as does `README.md`. `ToolManager` also implements list/exists/create/delete file actions; exposing them needs a `file` intent in `INTENT_TRIGGERS`, a handler branch, and an entry here.

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

All gitignored, along with `.env.txt`, `__pycache__/`, `assets/`, `.vscode/` and a stray `memory_data 2/`. `.claude/` is **not** ignored.

`utils.clean_temp_audio()` runs at startup and shutdown and deletes every `*.wav` in the system temp dir.

## Dependencies

`groq`, `python-dotenv` · `sounddevice`, `scipy`, `numpy<2.0` · `torch`, `transformers`, `accelerate`, `safetensors` · `edge-tts`, `pygame`, `langdetect` · `requests`, `beautifulsoup4`, `ddgs` · `pytest` · `pystray`, `Pillow`

The `torch` in `requirements.txt` is CPU-only on Windows; the CUDA build must be installed separately (see README).

## Conventions

- **Handlers return `bool`** — `True` to keep listening, `False` to shut down. Only `terminate` returns `False`.
- **Tool methods return a dict** with `success` plus either result fields or `error`. Callers check `result['success']` before reading anything else.
- **Failures are caught and reported, not raised** — a failed command should never kill the listening loop.
- **Regexes that run per sentence or per chunk are compiled at module level**, not inside the function.
- Comments are sparse by design: explain *why* (the ordering constraint in `_extract_all_content`, the md5 rationale), not *what*.

## Testing

pytest, configured by `pytest.ini` (`pythonpath = .`, `testpaths = tests`).

```
python -m pytest                      unit tests (IntentDetector only)
python tests/snapshot_intents.py      compare 49 sample utterances against tests/intent_baseline.json
python tests/snapshot_intents.py --update   re-record the baseline — only for an intended routing change
```

The snapshot pins current routing, it does not prove it correct. `tests/test_intent_detector.py` carries explanatory teaching comments; new tests should follow the house style and stay comment-free.

Everything outside `IntentDetector` is untested. When changing routing, parsing, or extraction logic, verify against the real behaviour rather than by inspection — stub the services with `MagicMock` and assert on what each tool method was called with.

## Possible Enhancements

- A `file` intent exposing `ToolManager`'s list/exists/create/delete actions
- Lightweight local wake-word model (currently full Whisper on every 1.5s window)
- Continuous conversation mode — skip re-triggering the wake word for follow-ups
- Vector embeddings for semantic memory instead of word overlap
- Pruning of old episodic memories (nothing deletes them today)
