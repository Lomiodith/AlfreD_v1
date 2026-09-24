import os
import sys

from dotenv import load_dotenv

# Frozen by PyInstaller, bundled files unpack to a temp dir (_MEIPASS) that is
# deleted on exit, so anything written at runtime goes next to the executable.
if getattr(sys, "frozen", False):
    BUNDLE_DIR = sys._MEIPASS
    RUNTIME_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BUNDLE_DIR, ".env"))


def _str(name, default):
    return os.getenv(name, default)


def _float(name, default):
    return float(os.getenv(name, default))


def _int(name, default):
    return int(os.getenv(name, default))


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Downloaded models live outside the project (and off the system drive).
# Unset, the libraries fall back to their default caches in the user profile.
MODELS_DIR = os.getenv("MODELS_DIR")
HF_CACHE_DIR = os.path.join(MODELS_DIR, "huggingface", "hub") if MODELS_DIR else None
TORCH_HUB_DIR = os.path.join(MODELS_DIR, "torch", "hub") if MODELS_DIR else None
# Also for libraries that download without a cache_dir option (Kokoro's voices).
# Only this process is affected, and huggingface_hub reads it at import time,
# which is why the entry points import config first.
if HF_CACHE_DIR:
    os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)

WAKE_WORD = _str("WAKE_WORD", "Alfred")
WAKE_WORD_DURATION = _float("WAKE_WORD_DURATION", 1.5)

SAMPLE_RATE = 16000
SILENCE_SECONDS = _float("SILENCE_SECONDS", 2.0)
MAX_RECORDING_SECONDS = _float("MAX_RECORDING_SECONDS", 20)
# Smart Turn: at a short pause, a small model judges from the audio whether the
# speaker is finished, ending the turn without waiting out SILENCE_SECONDS.
SMART_TURN = _bool("SMART_TURN", False)
SMART_TURN_PAUSE_SECONDS = _float("SMART_TURN_PAUSE_SECONDS", 0.5)
SMART_TURN_THRESHOLD = _float("SMART_TURN_THRESHOLD", 0.5)
# After Alfred asks a question, how long to wait for a reply without the wake word.
FOLLOW_UP_SECONDS = _float("FOLLOW_UP_SECONDS", 5)
STT_MODEL = _str("STT_MODEL", "distil-whisper/distil-large-v3.5")
# Languages you speak (ISO codes). A transcript that seems to be in another one
# makes Alfred ask whether that was intended, instead of answering it.
ALLOWED_LANGUAGES = [c.strip() for c in _str("ALLOWED_LANGUAGES", "en,ro").split(",")]

# Models tried in order, each as provider:model with provider "groq" or
# "llamacpp" (local). The next one is used when a model is rate-limited or fails.
LLM_CHAIN = [
    entry.strip()
    for entry in _str(
        "LLM_CHAIN", "groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b"
    ).split(",")
    if entry.strip()
]
# llama.cpp's llama-server; with MTP (multi-token prediction) models,
# --spec-type draft-mtp makes the first sentence 3-5x faster. AlfreD starts
# it when a llamacpp model is in LLM_CHAIN and it isn't already running.
LLAMACPP_URL = _str("LLAMACPP_URL", "http://127.0.0.1:8080")
LLAMACPP_SERVER = _str("LLAMACPP_SERVER", "")
LLAMACPP_MODEL = _str("LLAMACPP_MODEL", "")
LLAMACPP_ARGS = _str(
    "LLAMACPP_ARGS",
    "-ngl 999 -c 16384 --parallel 1 -fa on -ctk q8_0 -ctv q8_0 -t 8 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 6",
).split()
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.3)
# How much models think before answering, per output mode: low / medium / high.
# gpt-oss honours the level; Qwen can only think or not, and even its "low"
# costs 10+ s, so for Qwen anything below medium means no thinking.
LLM_REASONING_VOICE = _str("LLM_REASONING_VOICE", "low")
LLM_REASONING_TEXT = _str("LLM_REASONING_TEXT", "medium")
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1024)
# Text mode shows full answers, so it needs room for long ones.
LLM_MAX_TOKENS_TEXT = _int("LLM_MAX_TOKENS_TEXT", 4096)

TTS_VOICE = _str("TTS_VOICE", "en-GB-RyanNeural")
# "edge" (online, all languages) or "kokoro" (local and faster, English only;
# other languages still go to edge-tts).
TTS_ENGINE = _str("TTS_ENGINE", "edge")
KOKORO_VOICE = _str("KOKORO_VOICE", "bm_george")
# cuda: ~0.15 s per sentence for ~0.3 GB of GPU memory. cpu frees that memory
# but takes ~1.3 s per sentence, slower than edge-tts.
KOKORO_DEVICE = _str("KOKORO_DEVICE", "cuda")
# Speak answers (short, "Want more?") or only print them (full length). Toggle
# at runtime with "voice mode" / "text mode".
VOICE_OUTPUT = _bool("VOICE_OUTPUT", True)

EMBEDDING_MODEL = _str(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MEMORY_MIN_SIMILARITY = _float("MEMORY_MIN_SIMILARITY", 0.35)

# Folders searched when the user asks to open a file by name, separated by ";"
# on Windows (":" elsewhere), since folder names can contain commas.
FILE_SEARCH_DIRS = [
    d
    for d in (
        os.path.normpath(os.path.expanduser(entry.strip()))
        for entry in (
            os.getenv("FILE_SEARCH_DIRS")
            or os.pathsep.join(["~/Desktop", "~/Documents", "~/Downloads", RUNTIME_DIR])
        ).split(os.pathsep)
        if entry.strip()
    )
    if os.path.isdir(d)
]

# OAuth client and token; keep them out of the project folder (it syncs to Google Drive).
SECRETS_DIR = _str("SECRETS_DIR", os.path.join(RUNTIME_DIR, "secrets"))
GOOGLE_CREDENTIALS_FILE = os.path.join(SECRETS_DIR, "google_credentials.json")
GOOGLE_TOKEN_FILE = os.path.join(SECRETS_DIR, "google_token.json")

LOG_LEVEL = _str("LOG_LEVEL", "INFO")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")
