import asyncio
import logging
import os
import queue
import re
import sys
import tempfile
import threading
import time

import edge_tts
import numpy as np
import pygame
from langdetect import detect, LangDetectException
from scipy.io.wavfile import write as write_wav

from config import KOKORO_DEVICE, KOKORO_VOICE, TTS_ENGINE
from utils import stop_event

logger = logging.getLogger(__name__)

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
    (re.compile(r"【[^】]*】"), ""),  # gpt-oss citation marks 【4†L1-L4】
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


KOKORO_SAMPLE_RATE = 24000
MIN_WORDS_TO_DETECT = 4


class TTSService:
    """Speech output as a two-stage pipeline: a synthesis worker turns queued
    sentences into audio files while a playback worker plays the previous one,
    so there is no pause between sentences for the next one to be synthesised."""

    def __init__(self, default_voice="en-GB-RyanNeural"):
        self.default_voice = default_voice
        self.interrupted = False
        # Bumped on every stop/reset; queued items from an older generation are
        # dropped, so an interrupted answer can't resume speaking later.
        self._generation = 0
        self._text_queue: "queue.Queue" = queue.Queue()
        self._audio_queue: "queue.Queue" = queue.Queue()
        self._kokoro = self._load_kokoro() if TTS_ENGINE == "kokoro" else None
        self._last_language = "en"
        detect("Warm up the language detector.")  # loads its profiles (~0.4 s)
        pygame.mixer.init()
        self._workers = [
            threading.Thread(target=self._synthesis_worker, daemon=True),
            threading.Thread(target=self._playback_worker, daemon=True),
        ]
        for worker in self._workers:
            worker.start()
        logger.info(f"🔊 TTS Service initialized (default voice: {default_voice})")

    def say(self, text):
        """Queue a sentence and return immediately."""
        if not text or not text.strip() or self.interrupted or stop_event.is_set():
            return
        clean_text = self._clean_for_speech(text)
        if clean_text and self._is_speakable(clean_text):
            self._text_queue.put((self._generation, clean_text))

    def wait_until_done(self):
        while (
            self._text_queue.unfinished_tasks or self._audio_queue.unfinished_tasks
        ) and not (self.interrupted or stop_event.is_set()):
            time.sleep(0.05)

    def speak(self, text):
        self.say(text)
        self.wait_until_done()

    def stop(self):
        self.interrupted = True
        self._generation += 1
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def reset(self):
        self._generation += 1
        self.interrupted = False
        _flush_input()

    def shutdown(self):
        self.stop()
        self._text_queue.put(None)
        self._audio_queue.put(None)
        for worker in self._workers:
            worker.join(timeout=2)
        try:
            pygame.mixer.quit()
        except Exception:
            pass

    @staticmethod
    def _load_kokoro():
        # Imported here: Kokoro needs transformers >= 5, and AlfreD must still
        # run on edge-tts without it.
        from kokoro import KPipeline

        # The voice name's first letter is Kokoro's language code: b = British.
        kokoro = KPipeline(
            lang_code=KOKORO_VOICE[0],
            repo_id="hexgrad/Kokoro-82M",
            device=KOKORO_DEVICE,
        )
        list(kokoro("warm up", voice=KOKORO_VOICE))
        # Kokoro's phonemizer warns "words count mismatch" on harmless input, in
        # the middle of the spoken answer. Its logger only exists (with its own
        # level and handler) once the warm-up above has run.
        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        logger.info(f"🗣️ Kokoro loaded on {KOKORO_DEVICE} (voice {KOKORO_VOICE})")
        return kokoro

    def _language(self, text):
        # langdetect guesses wildly on a few words ("Want more?" came back as
        # non-English), so short phrases keep the previous sentence's language.
        if len(text.split()) < MIN_WORDS_TO_DETECT:
            return self._last_language
        try:
            self._last_language = detect(text).split("-")[0]
        except LangDetectException:
            pass
        return self._last_language

    def _synthesize(self, text, loop) -> str:
        """Write `text` as audio to a temp file and return its path."""
        language = self._language(text)
        if self._kokoro and language == "en":
            audio = np.concatenate(
                [r.audio.numpy() for r in self._kokoro(text, voice=KOKORO_VOICE)]
            )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            write_wav(path, KOKORO_SAMPLE_RATE, (audio * 32767).astype(np.int16))
            return path

        voice = VOICE_MAP.get(language, self.default_voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        loop.run_until_complete(edge_tts.Communicate(text, voice).save(path))
        return path

    def _clean_for_speech(self, text):
        for pattern, replacement in SPEECH_SUBSTITUTIONS:
            text = pattern.sub(replacement, text)
        return text.translate(QUOTE_TRANSLATION).strip()

    def _is_speakable(self, text):
        return len(NON_LETTERS.sub("", text)) >= 2

    def _synthesis_worker(self):
        loop = asyncio.new_event_loop()
        while (item := self._text_queue.get()) is not None:
            generation, text = item
            try:
                if generation == self._generation:
                    self._audio_queue.put((generation, self._synthesize(text, loop)))
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                self._text_queue.task_done()
        loop.close()

    def _playback_worker(self):
        while (item := self._audio_queue.get()) is not None:
            generation, path = item
            try:
                if generation == self._generation:
                    self._play(path)
            except Exception as e:
                logger.error(f"Playback error: {e}")
            finally:
                _remove_quietly(path)
                self._audio_queue.task_done()

    def _play(self, path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        try:
            while pygame.mixer.music.get_busy():
                if stop_event.is_set():
                    self.stop()
                    return
                if _kbhit():
                    _getch()
                    self.stop()
                    logger.info("\n⏹️ Speech interrupted.")
                    return
                time.sleep(0.05)
        finally:
            pygame.mixer.music.unload()


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass
