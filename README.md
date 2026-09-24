# AlfreD

A voice-activated AI assistant that runs locally on your machine. Say **"Alfred"**, ask a question, and it answers out loud — with web search, file access, web scraping, and a memory of past conversations.

Speech recognition runs locally (Whisper). The language model runs on Groq. Text-to-speech uses Microsoft Edge voices and auto-detects the language you're speaking.

---

## Requirements

- **Python 3.12** (3.8+ works, 3.12 is what this is developed against)
- A **microphone**
- An **internet connection** (for the LLM, search, and TTS)
- A **Groq API key** — free at [console.groq.com](https://console.groq.com)
- *Optional:* an NVIDIA GPU for much faster transcription

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install PyTorch for your hardware

The `torch` in `requirements.txt` is **CPU-only on Windows**. Transcription will work but is noticeably slower.

**NVIDIA GPU (Windows/Linux)** — install the CUDA build instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon** — the default wheel already includes MPS support; nothing extra to do.

AlfreD picks the best available device automatically and prints which one it chose at startup:

```
🔧 Using device: cuda
```

### 3. Add your API key

Create a file called **`.env.txt`** in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

That's the only key required. This file is gitignored — don't commit it.

### 4. First run

The first launch downloads the Whisper model (~1.5 GB) and the voice-activity model. This happens once; later starts are fast.

---

## Running

**System tray (recommended)**

```bash
python alfred_tray.py
```

Adds an icon to your system tray with **Start Alfred** / **Stop Alfred** / **Quit**. AlfreD only listens while started, so you can leave the tray app running all day.

**Console**

```bash
python AlfreD_v1.py
```

Starts listening immediately. `Ctrl+C` to quit.

---

## Talking to AlfreD

It's always a **two-step** exchange:

1. Say **"Alfred"** — it listens in 1.5-second windows for the wake word
2. Wait for `✅ Wake word detected!`, then speak your command

It records until you **stop talking** (up to 20 seconds), so just speak naturally and pause when you're done. No need to rush.

```
Listening for your calling, master...
Heard: "Alfred"
✅ Wake word detected! Listening for command...
Command heard: "what is the capital of France"
🔥 Alfred: The capital of France is Paris.
```

**To interrupt AlfreD mid-sentence, press any key.** Speech stops immediately.

---

## Commands

You don't need exact phrasing — each capability has several trigger phrases, and close matches are accepted (helpful when speech recognition mishears you). Anything that doesn't match a trigger is treated as normal conversation.

### Just talk

Ask anything. AlfreD remembers the conversation, so follow-ups work naturally.

> "Alfred, explain how a jet engine works"
> "Alfred, tell me more about that"

### Search the web

Say: `search for` · `look up` · `find me` · `google` · `search` · `look for` · `what is` · `who is` · `tell me about` · `find out`

> "Alfred, search for the best hiking boots"
> "Alfred, what is quantum entanglement"

Results are cached for an hour, so repeating a search is instant and free.

### Run a system command

Say: `run command` · `execute` · `terminal` · `shell command`

> "Alfred, run command git status"
> "Alfred, execute ls -la | head"

Commands run through **bash**, so pipes, `&&` and Unix tools work. On Windows this is Git Bash, found automatically from your Git for Windows install; without it, system commands report an error. Runs in safe mode — see [Safety](#safety) below.

### Read a file

Say: `read file` · `open file` · `show file` · `read the file` · `what's in file`

> "Alfred, read file config.py"

### Write a file

Say: `write file` · `save file` · `create file` · `write to file` — then **`with content`**

> "Alfred, write file notes.txt with content remember to buy milk"

The words *"with content"* are what separate the path from the text.

### Scrape a web page

Say: `scrape` · `web scrape` · `get the page` · `fetch the page` · `grab the page`

Add **`for <type>`** to choose what to extract — `text` (default), `links`, `images`, `metadata`, or `all`.

> "Alfred, scrape https://example.com"
> "Alfred, web scrape https://example.com for links"

### Housekeeping

| What you want | Say |
|---|---|
| See the command reference | `show commands` · `what can you do` |
| Forget the conversation so far | `clear context` · `start fresh` · `forget everything` · `new conversation` |
| See timing statistics | `performance stats` · `show stats` · `performance` |
| Shut AlfreD down | `close script` · `shut down` · `goodbye` · `turn off` · `exit` · `quit` |

---

## Memory

AlfreD keeps two kinds of memory, both stored locally in `memory_data/`:

- **Conversation memory** — past exchanges in a SQLite database. Relevant ones are pulled back in automatically when they relate to what you're asking.
- **Preferences** — it notices patterns over time (whether you prefer brief or detailed answers, which topics you return to, how formally you speak) and adapts.

When a conversation gets long, older messages are automatically summarized so context isn't lost but stays within the model's limit. Say `clear context` to wipe the current conversation; delete the `memory_data/` folder to erase everything permanently.

---

## Configuration

Edit `config.py`:

| Setting | Default | What it does |
|---|---|---|
| `WAKE_WORD` | `"Alfred"` | The word that wakes it up |
| `WAKE_WORD_DURATION` | `1.5` | Seconds per wake-word listening window |
| `TEMPERATURE` | `0.3` | Response creativity — higher is more varied |
| `SAMPLE_RATE` | `16000` | Microphone sample rate (Whisper expects 16 kHz) |
| `WHISPER_MODEL_ID` | `distil-whisper/distil-large-v3.5` | Speech recognition model |

**Voice** — set in `AlfreD_v1.py`:

```python
tts_service = TTSService(default_voice="en-GB-RyanNeural")
```

AlfreD detects the language of each sentence and switches voices automatically (17 languages supported — see `VOICE_MAP` in `tts_service.py`). The default voice is used when detection is uncertain.

**Models** — set in `llm_service.py`. Primary is `openai/gpt-oss-120b`, falling back to `llama-3.3-70b-versatile` if it fails or is rate-limited.

**Recording sensitivity** — `record_until_silence()` in `audio_processor.py` takes `max_duration` (default 20s) and `silence_threshold` (default 2.0s of quiet before it stops). Raise the threshold if it cuts you off while you're thinking.

---

## Safety

System commands and file operations run in **safe mode** by default:

- Destructive commands are blocked (`rm`, `del`, `format`, `mkfs`, `shutdown`, `reboot`, `fdisk`, `chmod 777`, …)
- Unix system directories are off limits for file operations (`/etc`, `/sys`, `/proc`, `/dev`, `/boot`, `/root`). Windows paths are not restricted
- Commands time out after 30 seconds
- Scraping is restricted to `http`/`https` and blocks localhost and loopback addresses

Safe mode is set where the tool manager is constructed in `conversation_handler.py`:

```python
self.tool_manager = ToolManager(safe_mode=True)
```

Turning it off removes every check above. Don't, unless you have a specific reason.

---

## Files AlfreD creates

| Path | Contents | Safe to delete? |
|---|---|---|
| `memory_data/memory.db` | Conversation history | Yes — erases memory |
| `memory_data/user_preferences.json` | Learned preferences | Yes — resets learning |
| `search_cache/` | Cached search results (1 hour) | Yes |

All three are gitignored.

---

## Troubleshooting

**It never hears the wake word.** Check your default input device and that the mic isn't muted. `Heard: "..."` prints whatever it transcribed each cycle — if that's empty or garbage, it's an input problem, not a recognition one. A quiet mic transcribes to nothing.

**Transcription is slow.** You're likely on CPU. Check the `🔧 Using device:` line at startup; if it says `cpu` and you have an NVIDIA card, install the CUDA build of torch (see [Setup](#2-install-pytorch-for-your-hardware)).

**It cuts me off mid-sentence.** Raise `silence_threshold` in `record_until_silence()`.

**"I'm having trouble with the model."** Both the primary and fallback model failed — usually a missing/invalid `GROQ_API_KEY` or a rate limit. The console shows the underlying error.

**No sound.** Edge TTS needs internet. Very short or symbol-only replies are skipped deliberately.

**It misroutes what I say.** Trigger phrases win over conversation, so "what is..." is always treated as a search. Phrase it differently if you want a plain answer — or edit `INTENT_TRIGGERS` in `intent_detector.py`.

---

## Project layout

```
AlfreD_v1.py            Entry point and main loop
alfred_tray.py          System tray wrapper
config.py               Settings and API keys

audio_processor.py      Recording, voice-activity detection, Whisper
conversation_handler.py Command routing and conversation flow
intent_detector.py      Maps spoken phrases to capabilities
llm_service.py          Groq client, streaming, model fallback
tts_service.py          Speech output and language detection

search_service.py       Web search with caching
tool_manager.py         Commands, files, scraping (with safety checks)
memory_manager.py       Conversation memory and preference learning
performance_monitor.py  Operation timing
commands.py             Command reference text
utils.py                Temp file cleanup
```
