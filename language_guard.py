"""Spots transcripts in a language the user doesn't speak.

Multilingual speech models (Parakeet) guess the language themselves and, on a
word or two, sometimes guess wrong: "Alfred." came back as Cyrillic "Альфред.".
Rather than send that to the LLM, Alfred asks whether it was intended.
"""

from typing import Iterable, Optional

from langdetect import DetectorFactory, LangDetectException, detect_langs

from config import WAKE_WORD

DetectorFactory.seed = 0  # langdetect is otherwise non-deterministic

CONFIDENCE = 0.9
# Fewer Latin-script words is too little to judge: "Okay." came back Tagalog,
# "so" Danish. A missed misfire only costs "please repeat" from the model; a
# false alarm interrupts a perfectly good request.
MIN_WORDS = 3
LANGUAGE_NAMES = {
    "bg": "Bulgarian",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
}


def _is_latin(char: str) -> bool:
    return "a" <= char.lower() <= "z" or "À" <= char <= "ɏ"


def _detect(text: str):
    """(language code, confident?) for `text` without the wake word, which is
    a name and no evidence of any language ("Alfred, so" came back Danish)."""
    words = [w for w in text.split() if w.strip(".,!?").lower() != WAKE_WORD.lower()]
    try:
        top = detect_langs(" ".join(words))[0]
    except LangDetectException:
        return None, False
    return top.lang, len(words) >= MIN_WORDS and top.prob >= CONFIDENCE


def spoken_language(text: str) -> Optional[str]:
    """Name of the language `text` is in, when langdetect is confident."""
    code, confident = _detect(text)
    return LANGUAGE_NAMES.get(code) if confident else None


def unexpected_language(text: str, allowed: Iterable[str]) -> Optional[str]:
    """Return the name of the language `text` appears to be in when that isn't
    one of `allowed` (ISO codes); None when it's fine or too short to judge.
    Languages without a name here are never reported: asking "did you mean to
    speak da?" helps no one."""
    code, confident = _detect(text)
    non_latin = any(c.isalpha() and not _is_latin(c) for c in text)
    if code not in allowed and code in LANGUAGE_NAMES and (non_latin or confident):
        return LANGUAGE_NAMES[code]
    return None
