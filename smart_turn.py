"""End-of-turn detection with Pipecat's Smart Turn v3 (github.com/pipecat-ai/smart-turn).

At a short pause the recorder asks whether the speaker sounds finished, judged
from the audio itself (falling intonation vs. a trailing "and..."), instead of
waiting out a fixed silence.
"""

import logging

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

from config import HF_CACHE_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)

REPO_ID = "pipecat-ai/smart-turn-v3"
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
WINDOW_SECONDS = 8


class SmartTurn:
    def __init__(self):
        path = hf_hub_download(REPO_ID, MODEL_FILE, cache_dir=HF_CACHE_DIR)
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(path, sess_options=options)
        self._features = WhisperFeatureExtractor(chunk_length=WINDOW_SECONDS)
        logger.info(f"🗣️ Smart Turn loaded ({MODEL_FILE})")

    def completion_probability(self, audio: np.ndarray) -> float:
        """`audio`: float32 mono at 16 kHz, the current turn so far."""
        window = audio[-WINDOW_SECONDS * SAMPLE_RATE :]
        features = self._features(
            window,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SECONDS * SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        return float(self._session.run(None, {"input_features": features})[0][0].item())
