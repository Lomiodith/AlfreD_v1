import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import EMBEDDING_MODEL, HF_CACHE_DIR

logger = logging.getLogger(__name__)

MAX_TOKENS = 256


class Embedder:
    """Sentence embeddings via mean pooling, L2-normalised so a dot product is cosine similarity."""

    def __init__(self, model_id=EMBEDDING_MODEL):
        logger.info(f"🧬 Loading embedding model '{model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
        self.model = AutoModel.from_pretrained(model_id, cache_dir=HF_CACHE_DIR).eval()

    def embed(self, texts: List[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return F.normalize(pooled, dim=1).numpy().astype(np.float32)
