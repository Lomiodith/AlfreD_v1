# AlfreD - Voice-Activated AI Assistant

## Project Overview

**AlfreD** is a voice assistant in Python. It listens for a wake word, transcribes locally (Parakeet or Whisper), and lets an LLM answer and call tools (web search, Gmail, Calendar, GitHub, timers, memory, bash, files) through a tool-calling loop, speaking the reply sentence by sentence.

User-facing documentation lives in `README.md`. This file is the internal map.

## Architecture

```
alfred_tray.py ─► AlfreD_v2.py (main loop: announce timers → wake word → command)
                        │
                 conversation_handler.py  (conversation state, voice/text mode,
                        │                   follow-ups, language guard)
   ┌──────────┬─────────┼──────────┬─────────────┬────────────┐
audio_      intent_   llm_       tts_          tools.py      memory_manager
processor   detector  service    service       (ToolRegistry) + embeddings
+ smart_turn          (model     (Kokoro/Edge,  core_tools,
                       chain +    pipelined)    memory_tools,
                       agent loop)              github_tools,
                                                google_tools
```

`ConversationHandler` owns the history and wires everything; services are built in `AlfreD_v2.main()`.

## Request flow

1. **Wake:** `record_audio` (1.5 s) → Silero VAD says speech? → STT → `WAKE_WORD` substring. Silent windows never reach STT (it hallucinates "Thank you." on silence).
2. **Command:** `record_until_silence` → Silero per 32 ms chunk; at a `SMART_TURN_PAUSE_SECONDS` pause, Smart Turn judges whether the turn is finished; `SILENCE_SECONDS` stays the backstop. Returns whether speech was heard; no speech → no transcription.
3. **Routing** (`_handle_command`): a pending language question is resolved first; then control intents (whole utterance only); then `language_guard.unexpected_language` may ask "Did you mean to speak X?"; everything else goes to `_dispatch_command`.
4. **LLM turn** (`_dispatch_command`): history + current date/time + voice-or-text style + retrieved memories + the user turn → `llm_service.run_agent` → sentences streamed to `tts_service.say` → `wait_until_done`.
5. **After:** store the exchange in memory; the conversation stays open: listen `FOLLOW_UP_SECONDS` for the next request without the wake word, after every answer (waiting only after a reply ending in `?` closed threads after one follow-up). Silence or `decline` ends it. `VOICE_STYLE` asks for a topic-specific follow-up question, never a generic "Want more?"

## Core Components

### Entry points
**`AlfreD_v2.py`**: imports `config` **first** (it sets `HF_HUB_CACHE` to `MODELS_DIR` before any Hugging Face library loads). Builds services, loops, cleans up in `finally`. Init failures are logged with a traceback.

**`alfred_tray.py`**: pystray wrapper on a daemon thread. "Running" is read from the thread itself (`_is_running`), so Start can't launch a second instance.

**Stopping**: `utils.stop_event` is polled by every blocking stage (recorders, LLM stream, TTS workers, handler). Only an in-progress transcription or model load can't be cut short.

**Key-press interrupt**: `tts_service._play` polls the keyboard; a key calls `tts.stop()`, and `run_agent(cancelled=lambda: tts.interrupted)` then ends streaming and runs no further tools. The spoken part goes into the history and Alfred listens again without the wake word. A voice "stop" was tried and dropped: transcribing the microphone during answers pushed the GPU over 12 GB and stalled llama-server, and over speakers the echo drowned the word

### `conversation_handler.py`
- `SYSTEM_PROMPT` (shared rules: reply in the user's language, ask to repeat garbled input, tool and memory use) plus **`VOICE_STYLE` / `TEXT_STYLE`, added per request, not stored**, so switching mode takes effect immediately
- **Retrieved memories go out with one request only, never into `conversation_history`.** Tool calls do go into the history (`AgentResult.trail`), with results trimmed to `HISTORY_TOOL_RESULT_CHARS`. Without them, the local model saw only "I've opened it" in past turns and started answering that way without calling the tool; untrimmed, every later request would resend every old search result. The trim happens once the next turn is stored, so the latest turn keeps its results whole and a "yes" to the follow-up question has the rest to continue from (trimmed immediately, the model had nothing left and narrated its confusion aloud). `auto_summarize_context` cuts only at a user message, since a tool result separated from its call is rejected
- `auto_summarize_context` collapses old turns above `MAX_CONTEXT_TOKENS`
- Text mode prints the finished answer rather than streaming: the sentence splitter drops the line breaks markdown needs
- `AgentResult.completed == False` (a round failed, e.g. all models rate-limited) → spoken apology; the partial text isn't stored
- `set_output_mode` is registered as a tool so natural phrasings ("switch to text so I can read it") work

### `intent_detector.py`
Only instant control commands. A command must be **the whole utterance** after stripping punctuation and `FILLER_WORDS` (applied to triggers too, or "what can you do" loses its "you"); fuzzy match on the whole utterance above `FUZZY_THRESHOLD`. `affirm` ("yes"/"da") is only acted on while a language question is pending; otherwise "yes" goes to the LLM (the answer to its follow-up question). `decline` ("no") and `stop` ("stop", "that's all") both end the conversation without an LLM call; while the language question is pending, `decline` answers it ("say it again") but `stop` still ends everything (as one intent, "stop" looped on that question). A reply made only of `NO_WORDS`/`STOP_WORDS` plus `STOP_GLUE` also matches ("No, I said okay, stop."). Input is transliterated from Cyrillic first (`CYRILLIC`): Parakeet writes a lone English "stop" as "Стоп.", and the language guard then asked "Did you mean Bulgarian?".

### `language_guard.py`
Flags a transcript as another language when it's non-Latin script, or langdetect is ≥ 0.9 confident on ≥ 3 words (wake word excluded) and the language isn't in `ALLOWED_LANGUAGES`. Exists because Parakeet guesses the language itself and misfires on one or two words ("Alfred." → "Альфред.").

### `llm_service.py`
- **Model chain** from `LLM_CHAIN` (`provider:model`, provider `llamacpp` or `groq`; malformed entries raise at startup). Currently: `llamacpp:qwythos-9b` → gpt-oss-120b → qwen3.8-27b (local first, under evaluation).
- **`llamacpp`** (`llamacpp_server.py`): AlfreD starts llama.cpp's `llama-server` (`LLAMACPP_SERVER`, `LLAMACPP_MODEL`, `LLAMACPP_ARGS`) unless one already answers, and stops only a server it started. On Windows the server it starts is put in a job object with kill-on-close (`_close_with_this_process`), so it also dies when AlfreD crashes or its console is closed; an orphan once held ~8 GB of GPU memory. A server that was already running is reused and left alone. The model is Qwythos-9B Q5_K_M MTP (Qwen 3.5 9B fine-tune, in `D:\AlfreD\models\qwythos`), fully on the GPU (`-ngl 999`). `--spec-type draft-mtp` uses its multi-token prediction head: 0.2-2.3 s to first speech vs 3-10 s for plain Qwen 3.5 9B. `--spec-draft-n-max 3`: measured on the RTX 4070 Super, 2-3 drafted tokens give ~97 tok/s, 6 gave 81, 8 gave 64, none 66 (longer drafts get rejected more). `-ub 2048` speeds prompt processing. `--parallel 1` so the 16K context isn't split across slots
- **Prompt-cache friendly requests:** everything that changes per turn (time, recalled memories, the reply-language note) is prepended to the *new user message*; the system prompt, style, tools and history stay byte-identical so llama.cpp (and Groq) reuse their prompt cache. `_single_system_message` folds all system messages into one at the start, because Qwen's template rejects a system message anywhere else
- **Recalled memories never include Alfred's past answers**, only the user's past questions and saved facts: auto-fed answers were repeated verbatim instead of searching, reinforcing a wrong answer every turn. The `recall` tool follows the same rule
- Each model has its own cooldown in `_cooldown_until`, set from Groq's `retry-after`; a cooling model is skipped
- **GPU budget (12 GB):** ~2-3.5 GB Windows and apps (depends on what's open), ~1.6 GB Parakeet fp16 + VAD, ~0.55 GB Kokoro, ~7.1 GB llama-server (Qwythos Q5 + 16K q8 KV); measured 11.3/12.3 GB with ~2 GB idle. Over budget, Windows spills GPU memory into RAM silently and the LLM runs 3-4× slower (5k-token prompts took 35 s)
- Groq client has `max_retries=0`: the SDK would otherwise sleep 20-40 s on a 429 in silence
- `_with_fallback(attempt, label, primary_attempts)`: the first model gets one retry for Groq's occasional "tool call validation failed"; rate limits skip straight on
- `run_agent`: stream a round, run requested tools one at a time (gpt-oss has no parallel calls), repeat, max `MAX_AGENT_STEPS`; the last round offers no tools. `_stream_step` accumulates streamed tool-call deltas by index
- **Thinking** per output mode (`LLM_REASONING_VOICE`/`_TEXT`), mapped per model in `_reasoning_kwargs`: gpt-oss gets the level; Groq's hybrid Qwen is on/off (`default`/`none`, off below medium) because even its "low" thinks for 10+ s. The local llama.cpp model thinks unless the level is `none`: with thinking off, Qwythos spoke its reasoning as the answer ("The user said yes, I should…"). llama-server returns the thinking in `reasoning_content`, which is never read; `--reasoning-budget 512` caps it. Cost: ~1.5 s per round (+3-4 s to first speech on a tool turn). `THINKING` strips inline `<think>` blocks from streamed content as a safety net (Groq Qwen returns them inline)

### `tools.py`, `github_tools.py`, `google_tools.py`, `timers.py`
`ToolRegistry.call` parses JSON arguments, catches every error into `{"error": ...}` and truncates to `MAX_RESULT_CHARS`. `core_tools` (search, bash, files, scrape, timers), `memory_tools` (remember/recall/forget): `remember` only on an explicit request; `recall` returns saved facts and past *questions*, never past answers. GitHub shells out to `gh` (read-only; disabled if `gh` is missing or logged out). Google: Gmail read-only, Calendar events; sign-in via `python google_tools.py`; token in `SECRETS_DIR`; disabled until signed in. Timers fire on threads but only **queue** an announcement that the main loop speaks, so it can't talk over an answer.

### `audio_processor.py` / `smart_turn.py`
STT via the generic `transformers` ASR pipeline from `STT_MODEL` (Parakeet needs transformers ≥ 5). `_is_speech` is the one Silero check; `vad_model.reset_states()` before each recording. `smart_turn.SmartTurn` runs Pipecat's Smart Turn v3.2 ONNX (8.7 MB, CPU, ~40 ms) on the last 8 s of audio via `WhisperFeatureExtractor`.

### `tts_service.py`
Two worker threads: synthesis (Kokoro for English when `TTS_ENGINE=kokoro`, else Edge voice from `VOICE_MAP`) and playback (pygame). `say()` queues, `wait_until_done()` blocks, `speak()` = both. `_generation` is bumped on stop/reset so stale queued sentences are dropped. Phrases under `MIN_WORDS_TO_DETECT` keep the previous sentence's language (langdetect guesses wildly on "Want more?"). `SPEECH_SUBSTITUTIONS` (compiled once, runs per sentence) strips markdown, URLs and gpt-oss citation marks `【…】`.

### `memory_manager.py` / `embeddings.py`
SQLite (`memory_data/memory.db`) with an `embedding` BLOB; `conversation_interaction` and `user_fact` rows are searchable (summaries aren't). A startup backfill embeds rows missing a vector; an in-memory matrix does cosine search (`search_memories`). `Embedder` = mean-pooled multilingual MiniLM, CPU. Always use `_connect()`; keep `MEMORY_COLUMNS` explicit (positional unpacking). `user_preference_learning` must keep its save.

### `search_service.py`
`ddgs` with `backend="auto"` (a meta-search across several engines; DuckDuckGo alone often answers 202). 5 results, snippets capped at `MAX_SNIPPET_CHARS`, since results are resent on every later round of a turn. md5 cache keys (built-in `hash()` is salted per process).

### `file_opener.py`
`open_file(name)`: a full path opens directly (a bare name never resolves against the working directory); otherwise `find` walks `FILE_SEARCH_DIRS` (`;`-separated, since folder names contain commas; capped at `SEARCH_SECONDS`, skipping hidden, `node_modules`, venv and similar folders) and keeps the best tier: exact name > name without extension > part of the name, with spaces, `_`, `-` and `.` ignored ("alfred tray" finds `alfred_tray.py`). Several matches go back to the model to ask. Before opening, the Windows association is read from the registry: if the default action would *run* the file (`.js` → WScript, `.bat`), a text file opens in VS Code instead and a program isn't opened at all

### `tool_manager.py`
Bash via Git Bash on Windows (`_find_bash` walks up from `git`, because PATH finds WSL's bash first). Safe mode: word-boundary blocklist on the whole string; `_is_path_safe` only knows Unix paths. `web_scraping` parses once; `_extract_all_content` runs text extraction last (it mutates the tree).

### `commands.py`
The instant-command section is **generated from `INTENT_TRIGGERS`**; `tests/test_commands.py` fails if an intent lacks a description.

### `config.py` / `.env`
Every setting comes from `.env` (template: `.env.example`). Frozen by PyInstaller: bundle files in `_MEIPASS`, runtime files (logs) next to the exe. `LOG_LEVEL=DEBUG` adds Smart Turn probabilities and every operation's timing.

## Environment & storage

- Runs on the **global Python 3.12** (upgraded to transformers 5.17 for Parakeet; the before/after `pip freeze` is in `D:\AlfreD\backups\`). CUDA torch 2.11
- **C: is nearly full.** Models live in `MODELS_DIR` (`D:\AlfreD\models`: Hugging Face cache, torch hub, `qwythos/` GGUF; llama.cpp itself in `D:\AlfreD\llama.cpp`). Secrets in `SECRETS_DIR` (`D:\AlfreD\secrets`), outside this Google-Drive-synced folder
- espeak-ng is installed system-wide (`C:\Program Files\eSpeak NG`); Kokoro also bundles its own
- `D:\AlfreD\wakeword\`: a trained openWakeWord-compatible `alfred.onnx` (never false-triggers, but under-confident on unseen voices, so not wired in) and its training venv

```
memory_data/   memory.db, user_preferences.json      (gitignored)
search_cache/  <md5>_<n>.json, 1 h TTL                (gitignored)
logs/          alfred.log, rotating                   (gitignored)
```

`.env`, `secrets/`, `__pycache__/`, `assets/`, `.vscode/` are gitignored; `.claude/` deliberately is not. `utils.clean_temp_audio()` deletes every `*.wav` in the temp dir at start and exit.

## Conventions

- **Handlers return `bool`**: `False` only for `terminate`
- **Tools return data** (dict/list/str); errors come back as `{"error": ...}` rather than raising
- **Failures are caught and logged**, never allowed to kill the listening loop
- Regexes that run per sentence or chunk are compiled at module level
- Formatted with **black**; comments explain *why*, not *what*
- Never read stored conversation text in `memory_data` when testing; use made-up data

## Testing

```
python -m pytest                          all unit tests
python tests/snapshot_intents.py          compare intent routing to tests/intent_baseline.json
python tests/snapshot_intents.py --update re-record, only for an intended routing change
```

Unit tests cover intents, the language guard, timers, the tool registry, memory (with a fake embedder and temp DB), the LLM chain/agent loop, Gmail text extraction and the command reference. Audio, TTS and live model behaviour aren't unit-tested: verify them with stubbed services and a real Groq call, or a live run with `LOG_LEVEL=DEBUG`.

## Open items

- Parakeet vs Whisper still under evaluation (Parakeet: Romanian and speed; Whisper: steadier on very short phrases)
- Local 9B models (Qwythos, Qwen 3.5) invent facts and once claimed a timer without calling `set_timer`; Qwythos is first in the chain while being evaluated. The system prompt now forbids claiming unperformed actions; Qwen 3.6 35B-A3B (partly in RAM) was tried and dropped as too slow and still unreliable
- Groq free-tier rate limits are the main reliability problem in heavy use
- Memory pruning: nothing deletes old episodic memories
