import os
import sys
from dotenv import load_dotenv

if getattr(sys, "frozen", False):
    _base = sys._MEIPASS
else:
    _base = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(_base, ".env.txt"))

SAMPLE_RATE = 16000
WAKE_WORD = "Alfred"
WAKE_WORD_DURATION = 1.5
TEMPERATURE = 0.3
WHISPER_MODEL_ID = "distil-whisper/distil-large-v3.5"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
