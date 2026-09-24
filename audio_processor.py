import os
import tempfile
import warnings

import numpy as np
import sounddevice as sd
import torch
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from config import SAMPLE_RATE, WHISPER_MODEL_ID
from utils import stop_event

VAD_CHUNK_SIZE = 512
SPEECH_PROBABILITY_THRESHOLD = 0.5
SUPPRESSED_WARNINGS = (
    "return_token_timestamps",
    "chunk_length_s",
    "experimental",
    "forced_decoder_ids",
)


class AudioProcessor:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"🔧 Using device: {self.device}")

        self._suppress_warnings()
        self.vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.vad_model.to("cpu")
        self.asr_pipeline = self._load_whisper()
        self._warm_up_model()

    @staticmethod
    def _suppress_warnings():
        for category in (FutureWarning, UserWarning):
            warnings.filterwarnings("ignore", category=category)
        for message in SUPPRESSED_WARNINGS:
            warnings.filterwarnings("ignore", message=f".*{message}.*")

    def _load_whisper(self):
        try:
            print(f"🚀 Loading Whisper model '{WHISPER_MODEL_ID}' on {self.device}...")

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(self.device)
            model.eval()
            processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=self.device,
            )
            print(f"✅ Whisper model loaded successfully on {self.device}")
            return asr_pipeline

        except Exception as e:
            print(f"❌ Error initializing Whisper model: {e}")
            raise

    def _warm_up_model(self):
        temp_path = None
        try:
            print("🔥 Warming up model...")
            with self.create_temp_audio_file() as temp_file:
                temp_path = temp_file.name
            write(temp_path, SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2, dtype=np.int16))
            self.asr_pipeline(temp_path)
            print("✅ Warm-up complete - ready for fast transcription!")
        except Exception as e:
            print(f"⚠️ Warm-up failed (continuing anyway): {e}")
        finally:
            self.cleanup_temp_file(temp_path)

    def record_audio(self, filename, duration):
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

    def record_until_silence(self, filename, max_duration=20, silence_threshold=2.0):
        """Record until `silence_threshold` seconds of silence, capped at `max_duration`."""
        chunks = []
        silent_chunks = 0
        silent_chunks_needed = int(silence_threshold * SAMPLE_RATE / VAD_CHUNK_SIZE)
        max_chunks = int(max_duration * SAMPLE_RATE / VAD_CHUNK_SIZE)

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

                tensor = torch.from_numpy(audio_chunk.flatten())
                speech_prob = self.vad_model(tensor, SAMPLE_RATE).item()

                if speech_prob < SPEECH_PROBABILITY_THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if (
                    silent_chunks >= silent_chunks_needed
                    and len(chunks) > silent_chunks_needed
                ):
                    break

        if not chunks:
            return
        audio = np.concatenate(chunks)
        write(filename, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    def transcribe_audio(self, filepath):
        if not filepath or not os.path.exists(filepath):
            print(f"❌ Audio file not found: {filepath}")
            return ""

        try:
            result = self.asr_pipeline(filepath)
            return result["text"].strip() if result and "text" in result else ""
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

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
