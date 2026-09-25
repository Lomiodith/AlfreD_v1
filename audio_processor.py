import logging
import os
import tempfile
import warnings

import numpy as np
import sounddevice as sd
import torch
from scipy.io.wavfile import write
from transformers import pipeline

from config import (
    HF_CACHE_DIR,
    MAX_RECORDING_SECONDS,
    SAMPLE_RATE,
    SILENCE_SECONDS,
    SMART_TURN,
    SMART_TURN_PAUSE_SECONDS,
    SMART_TURN_THRESHOLD,
    STT_MODEL,
    TORCH_HUB_DIR,
)
from utils import stop_event

logger = logging.getLogger(__name__)

VAD_CHUNK_SIZE = 512
SPEECH_PROBABILITY_THRESHOLD = 0.5
SUPPRESSED_WARNINGS = (
    "return_token_timestamps",
    "chunk_length_s",
    "experimental",
    "forced_decoder_ids",
)


def _to_chunks(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE / VAD_CHUNK_SIZE)


class AudioProcessor:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logger.info(f"🔧 Using device: {self.device}")

        self._suppress_warnings()
        if TORCH_HUB_DIR:
            torch.hub.set_dir(TORCH_HUB_DIR)
        self.vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.vad_model.to("cpu")
        self.asr_pipeline = self._load_stt()
        self.smart_turn = None
        if SMART_TURN:
            from smart_turn import SmartTurn

            self.smart_turn = SmartTurn()
        self._warm_up_model()

    @staticmethod
    def _suppress_warnings():
        for category in (FutureWarning, UserWarning):
            warnings.filterwarnings("ignore", category=category)
        for message in SUPPRESSED_WARNINGS:
            warnings.filterwarnings("ignore", message=f".*{message}.*")
        # Logged (not warned) after 10 pipeline calls: batching advice for bulk
        # jobs, meaningless when utterances arrive one at a time.
        logging.getLogger("transformers.pipelines.base").addFilter(
            lambda record: "pipelines sequentially" not in record.getMessage()
        )

    def _load_stt(self):
        """Any Hugging Face speech-recognition model: Whisper, or Parakeet
        (nvidia/parakeet-tdt-0.6b-v3, which needs transformers >= 5)."""
        try:
            logger.info(f"🚀 Loading speech model '{STT_MODEL}' on {self.device}...")
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=STT_MODEL,
                device=self.device,
                # Half precision halves Parakeet's 2.4 GB on the GPU, which it
                # shares with the local LLM; over 12 GB, Windows spills into RAM
                # and llama-server crawls at under a token per second.
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                model_kwargs={"cache_dir": HF_CACHE_DIR},
            )
            logger.info(f"✅ Speech model loaded on {self.device}")
            return asr_pipeline

        except Exception as e:
            logger.error(f"❌ Error loading speech model '{STT_MODEL}': {e}")
            raise

    def _warm_up_model(self):
        temp_path = None
        try:
            logger.info("🔥 Warming up model...")
            with self.create_temp_audio_file() as temp_file:
                temp_path = temp_file.name
            write(temp_path, SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2, dtype=np.int16))
            self.asr_pipeline(temp_path)
            logger.info("✅ Warm-up complete - ready for fast transcription!")
        except Exception as e:
            logger.warning(f"⚠️ Warm-up failed (continuing anyway): {e}")
        finally:
            self.cleanup_temp_file(temp_path)

    def _sounds_finished(self, chunks) -> bool:
        probability = self.smart_turn.completion_probability(
            np.concatenate(chunks).flatten()
        )
        logger.debug(f"Smart Turn: p(finished)={probability:.2f}")
        return probability >= SMART_TURN_THRESHOLD

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """`chunk`: VAD_CHUNK_SIZE float32 samples in [-1, 1]."""
        probability = self.vad_model(torch.from_numpy(chunk), SAMPLE_RATE).item()
        return probability >= SPEECH_PROBABILITY_THRESHOLD

    def record_audio(self, filename, duration) -> bool:
        """Record a fixed window; returns whether it contains any speech."""
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        if stop_event.wait(duration):
            sd.stop()
        sd.wait()
        write(filename, SAMPLE_RATE, recording)

        samples = recording.flatten().astype(np.float32) / 32768.0
        self.vad_model.reset_states()
        return any(
            self._is_speech(samples[start : start + VAD_CHUNK_SIZE])
            for start in range(0, len(samples) - VAD_CHUNK_SIZE + 1, VAD_CHUNK_SIZE)
        )

    def record_until_silence(
        self,
        filename,
        max_duration=MAX_RECORDING_SECONDS,
        silence_threshold=SILENCE_SECONDS,
        start_timeout=None,
    ) -> bool:
        """Record until `silence_threshold` seconds of silence after speech, capped
        at `max_duration`. Before any speech, give up after `start_timeout` seconds
        (default: `silence_threshold`). Returns whether speech was heard at all."""
        chunks = []
        silent_chunks = 0
        heard_speech = False
        silent_chunks_needed = _to_chunks(silence_threshold)
        start_chunks_needed = _to_chunks(start_timeout or silence_threshold)
        max_chunks = _to_chunks(max_duration)
        pause_chunks = _to_chunks(SMART_TURN_PAUSE_SECONDS) if self.smart_turn else -1
        self.vad_model.reset_states()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_CHUNK_SIZE,
        ) as stream:
            for _ in range(max_chunks):
                if stop_event.is_set():
                    break
                audio_chunk, _ = stream.read(VAD_CHUNK_SIZE)
                chunks.append(audio_chunk.copy())

                if self._is_speech(audio_chunk.flatten()):
                    silent_chunks = 0
                    heard_speech = True
                else:
                    silent_chunks += 1

                needed = silent_chunks_needed if heard_speech else start_chunks_needed
                if silent_chunks >= needed:
                    break
                # At a short pause, ask Smart Turn whether that was the end of
                # the sentence; the fixed silence above stays as the backstop.
                if (
                    heard_speech
                    and silent_chunks == pause_chunks
                    and self._sounds_finished(chunks)
                ):
                    break

        if not chunks:
            return False
        audio = np.concatenate(chunks)
        write(filename, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        return heard_speech

    def transcribe_audio(self, filepath):
        if not filepath or not os.path.exists(filepath):
            logger.error(f"❌ Audio file not found: {filepath}")
            return ""

        try:
            result = self.asr_pipeline(filepath)
            return result["text"].strip() if result and "text" in result else ""
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""
        finally:
            # PyTorch keeps freed GPU memory reserved for reuse; a 20 s command
            # grows that reserve, and llama-server needs the room.
            if self.device == "cuda":
                torch.cuda.empty_cache()

    @staticmethod
    def create_temp_audio_file():
        return tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    @staticmethod
    def cleanup_temp_file(filepath):
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
