import os
import sys
from dotenv import load_dotenv

if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
else:
    _base = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(_base, ".env.txt"))

SAMPLE_RATE = 16000
WAKE_WORD = "Alfred"
TERMINATION_PHRASE = "close script"
CLEAR_CONTEXT_PHRASE = "clear context"
TEMPERATURE = 0.3
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "e2e1906518dcf49c3")
WHISPER_MODEL_ID = "distil-whisper/distil-large-v3.5"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WAKE_WORD_DURATION = 1.5
COMMAND_DURATION = 10
