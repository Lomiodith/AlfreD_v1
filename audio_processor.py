import os
import tempfile
import warnings
import sounddevice as sd
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from config import SAMPLE_RATE, WHISPER_MODEL_ID


class AudioProcessor:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        print(f"🔧 Using device: {self.device}")

        self._suppress_warnings()
        self.model = None
        self.processor = None
        self.asr_pipeline = None
        self.vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.vad_model.to("cpu")
        self._initialize_model()

    def _suppress_warnings(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
        warnings.filterwarnings("ignore", message=".*chunk_length_s.*")
        warnings.filterwarnings("ignore", message=".*experimental.*")
        warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")

    def record_until_silence(self, filename, max_duration=20, silence_threshold=2.0):
        """Record audio until user stops talking.

        Args:
            filename: path to save the wav file
            max_duration: safety cap in seconds
            silence_threshold: seconds of silence before stopping
        """
        chunk_size = 512
        chunks = []
        silent_chunks = 0
        silent_chunks_needed = int(silence_threshold * SAMPLE_RATE / chunk_size)
        max_chunks = int(max_duration * SAMPLE_RATE / chunk_size)

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=chunk_size
        )

        stream.start()

        for _ in range(max_chunks):
            audio_chunk, _ = stream.read(chunk_size)
            chunks.append(audio_chunk.copy())

            tensor = torch.from_numpy(audio_chunk.flatten())
            speech_prob = self.vad_model(tensor, SAMPLE_RATE).item()

            if speech_prob < 0.5:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if (
                silent_chunks >= silent_chunks_needed
                and len(chunks) > silent_chunks_needed
            ):
                break

        stream.stop()
        stream.close()

        audio = np.concatenate(chunks)
        audio_int16 = (audio * 32767).astype(np.int16)
        write(filename, SAMPLE_RATE, audio_int16)

    def _initialize_model(self):
        try:
            print(f"🚀 Loading Whisper model '{WHISPER_MODEL_ID}' on {self.device}...")

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID, low_cpu_mem_usage=True, use_safetensors=True
            )
            self.model.to(self.device)
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device,
            )

            print(f"✅ Whisper model loaded successfully on {self.device}")
            self._warm_up_model()

        except Exception as e:
            print(f"❌ Error initializing Whisper model: {e}")
            raise

    def record_audio(self, filename, duration):
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        write(filename, SAMPLE_RATE, recording)

    def _warm_up_model(self):
        try:
            print("🔥 Warming up model...")
            dummy_audio = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.int16)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            write(temp_path, SAMPLE_RATE, dummy_audio)
            self.asr_pipeline(temp_path)
            self.cleanup_temp_file(temp_path)
            print("✅ Warm-up complete - ready for fast transcription!")
        except Exception as e:
            print(f"⚠️ Warm-up failed (continuing anyway): {e}")

    def transcribe_audio(self, filepath):
        if not filepath or not os.path.exists(filepath):
            print(f"❌ Audio file not found: {filepath}")
            return ""

        if not self.asr_pipeline:
            print("❌ Model not loaded")
            return ""

        try:
            result = self.asr_pipeline(filepath)
            return result["text"].strip() if result and "text" in result else ""
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

    def create_temp_audio_file(self):
        return tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    def cleanup_temp_file(self, filepath):
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
