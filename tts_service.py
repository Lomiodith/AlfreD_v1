import edge_tts
import asyncio
import tempfile
import os
import re
import msvcrt
import pygame
from langdetect import detect, LangDetectException

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
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)           # **bold**
        text = re.sub(r'\*(.+?)\*', r'\1', text)               # *italic*
        text = re.sub(r'#{1,6}\s*', '', text)                   # ### headings
        text = re.sub(r'`(.+?)`', r'\1', text)                  # `code`
        text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)  # - bullet points
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)         # [link](url)
        text = re.sub(r'https?://\S+', '', text)                # bare URLs
        text = re.sub(r'\|', ' ', text)                         # table pipes
        text = re.sub(r'-{3,}', '', text)                       # table separators ---
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # lone numbers (table row indices)
        text = re.sub(r'[*_~`]', '', text)                      # leftover markdown symbols
        text = text.replace('„', '"').replace('"', '"').replace('"', '"')  # normalize quotes
        text = re.sub(r'\s{2,}', ' ', text)                     # collapse whitespace
        return text.strip()

    def _is_speakable(self, text):
        stripped = re.sub(r'[^a-zA-Z\u00C0-\u024F\u0400-\u04FF\u4e00-\u9fff\uac00-\ud7af]', '', text)
        return len(stripped) >= 2

    def speak(self, text):
        if not text or not text.strip() or self.interrupted:
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
        while msvcrt.kbhit():
            msvcrt.getch()

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
            if msvcrt.kbhit():
                key = msvcrt.getch()
                self.stop()
                print("\n⏹️ Speech interrupted.")
                return
            await asyncio.sleep(0.05)
        pygame.mixer.music.unload()
