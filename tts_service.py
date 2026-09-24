import asyncio
import os
import re
import sys
import tempfile

import edge_tts
import pygame
from langdetect import detect, LangDetectException

from utils import stop_event

if sys.platform == "win32":
    import msvcrt

    def _kbhit():
        return msvcrt.kbhit()

    def _getch():
        return msvcrt.getch()

    def _flush_input():
        while msvcrt.kbhit():
            msvcrt.getch()

else:
    import select
    import termios
    import tty

    def _kbhit():
        if not sys.stdin.isatty():
            return False
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(dr)

    def _getch():
        if not sys.stdin.isatty():
            return b""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.encode() if isinstance(ch, str) else ch

    def _flush_input():
        if not sys.stdin.isatty():
            return
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass


VOICE_MAP = {
    "en": "en-GB-RyanNeural",
    "ro": "ro-RO-EmilNeural",
    "de": "de-DE-ConradNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "it": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural",
    "nl": "nl-NL-MaartenNeural",
    "pl": "pl-PL-MarekNeural",
    "hu": "hu-HU-TamasNeural",
    "tr": "tr-TR-AhmetNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "zh": "zh-CN-YunxiNeural",
    "ru": "ru-RU-DmitryNeural",
    "ar": "ar-SA-HamedNeural",
    "hi": "hi-IN-MadhurNeural",
}

# Applied in order to every streamed sentence, so they are compiled once.
SPEECH_SUBSTITUTIONS = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),  # **bold**
    (re.compile(r"\*(.+?)\*"), r"\1"),  # *italic*
    (re.compile(r"#{1,6}\s*"), ""),  # ### headings
    (re.compile(r"`(.+?)`"), r"\1"),  # `code`
    (re.compile(r"^\s*[-\u2022]\s*", re.MULTILINE), ""),  # bullet points
    (re.compile(r"\[(.+?)\]\(.+?\)"), r"\1"),  # [link](url)
    (re.compile(r"https?://\S+"), ""),  # bare URLs
    (re.compile(r"\|"), " "),  # table pipes
    (re.compile(r"-{3,}"), ""),  # table separators
    (re.compile(r"^\s*\d+\s*$", re.MULTILINE), ""),  # lone row numbers
    (re.compile(r"[*_~`]"), ""),  # leftover markdown
    (re.compile(r"\s{2,}"), " "),  # collapse whitespace
]

# Curly quotes confuse the voice; fold them onto their ASCII equivalents.
QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201e": '"',
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
)

NON_LETTERS = re.compile(
    r"[^a-zA-Z\u00C0-\u024F\u0400-\u04FF\u4e00-\u9fff\uac00-\ud7af]"
)


class TTSService:
    def __init__(self, default_voice="en-GB-RyanNeural"):
        self.default_voice = default_voice
        self.interrupted = False
        self.loop = asyncio.new_event_loop()
        pygame.mixer.init()
        print(f"🔊 TTS Service initialized (default voice: {default_voice})")

    def _detect_voice(self, text):
        try:
            lang = detect(text)
            lang_base = lang.split("-")[0]
            return VOICE_MAP.get(lang_base, self.default_voice)
        except LangDetectException:
            return self.default_voice

    def _clean_for_speech(self, text):
        for pattern, replacement in SPEECH_SUBSTITUTIONS:
            text = pattern.sub(replacement, text)
        return text.translate(QUOTE_TRANSLATION).strip()

    def _is_speakable(self, text):
        return len(NON_LETTERS.sub("", text)) >= 2

    def speak(self, text):
        if not text or not text.strip() or self.interrupted or stop_event.is_set():
            return
        try:
            clean_text = self._clean_for_speech(text)
            if clean_text and self._is_speakable(clean_text):
                voice = self._detect_voice(clean_text)
                self.loop.run_until_complete(self._speak_async(clean_text, voice))
        except Exception as e:
            print(f"TTS error: {e}")

    def stop(self):
        self.interrupted = True
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

    def reset(self):
        self.interrupted = False
        _flush_input()

    def shutdown(self):
        self.stop()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        try:
            self.loop.close()
        except Exception:
            pass

    async def _speak_async(self, text, voice):
        tmp_path = os.path.join(tempfile.gettempdir(), "alfred_tts.mp3")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if stop_event.is_set():
                self.stop()
                return
            if _kbhit():
                _getch()
                self.stop()
                print("\n⏹️ Speech interrupted.")
                return
            await asyncio.sleep(0.05)
        pygame.mixer.music.unload()
