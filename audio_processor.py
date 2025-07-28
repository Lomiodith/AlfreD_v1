import os
import tempfile
import warnings
import sounddevice as sd
import torch
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from config import SAMPLE_RATE, WHISPER_MODEL_ID


class AudioProcessor:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._suppress_warnings()
        self._initialize_model()

    def _suppress_warnings(self):
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper.generation_whisper")
        warnings.filterwarnings("ignore", message="You have passed task=transcribe, but also have set `forced_decoder_ids` to .*")
        warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*will be ignored in favor of task=transcribe.*")

    def _initialize_model(self):
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            print(f"Local Whisper model '{WHISPER_MODEL_ID}' initialized successfully on {self.device}.")
        except Exception as e:
            print(f"Error initializing local Whisper model: {e}")
            raise

    def record_audio(self, filename, duration):
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1, 
            dtype="int16"
        )
        sd.wait()
        write(filename, SAMPLE_RATE, recording)

    def transcribe_audio(self, filepath):
        if not filepath or not os.path.exists(filepath):
            print(f"Error: Audio file not found: {filepath}")
            return ""
        
        try:
            result = self.asr_pipeline(filepath, generate_kwargs={"language": "english"})
            return result["text"]
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def create_temp_audio_file(self):
        return tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    def cleanup_temp_file(self, filepath):
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass