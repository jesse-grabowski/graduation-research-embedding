# embedder_st.py
# SentenceTransformers utilities + embedding model configuration.
#
# Put all embedding-related knobs here so the pipeline script stays clean.

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ============================================================
# EMBEDDING MODEL CONFIG (edit here)
# ============================================================

# Model choice
MODEL_NAME: str = "Qwen/Qwen3-Embedding-4B"

# Embedding options
BATCH_SIZE: int = 64
NORMALIZE_EMBEDDINGS: bool = True

# Device override:
# - None => auto-detect (cuda > mps > cpu)
# - Or set to "cpu" / "mps" / "cuda"
FORCE_DEVICE: Optional[str] = None

# Optional: enable extra HF kwargs when running on CUDA.
# These are only applied if `device == "cuda"`.
ENABLE_CUDA_EXTRA_KWARGS: bool = True

# Passed through to SentenceTransformer(...) when CUDA is used.
# Note: These require a Transformers backend/model that supports them.
CUDA_MODEL_KWARGS: dict[str, Any] = {
    "attn_implementation": "flash_attention_2",
    "device_map": "auto",
}

# Passed through to SentenceTransformer(...) when CUDA is used.
TOKENIZER_KWARGS: dict[str, Any] = {
    "padding_side": "left",
}


# ============================================================
# Device selection
# ============================================================

def detect_best_device() -> str:
    """
    Choose best available device.
    Priority: CUDA > MPS > CPU
    Returns a device string understood by SentenceTransformers / torch.
    """
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def get_device() -> str:
    """Return FORCE_DEVICE if set, else auto-detect."""
    return FORCE_DEVICE or detect_best_device()


# ============================================================
# Model + tokenizer
# ============================================================

def build_embedder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[SentenceTransformer, Any, str]:
    """
    Build a SentenceTransformer model + its tokenizer.

    Args:
      model_name: overrides MODEL_NAME if provided
      device: overrides FORCE_DEVICE/auto-detect if provided

    Returns:
      (model, tokenizer, device_str)
    """
    model_name = model_name or MODEL_NAME
    device = device or get_device()

    extra_kwargs: dict[str, Any] = {}
    extra_kwargs["tokenizer_kwargs"] = TOKENIZER_KWARGS

    # Extra kwargs only when on CUDA (opt-in)
    if ENABLE_CUDA_EXTRA_KWARGS and device == "cuda":
        extra_kwargs["model_kwargs"] = CUDA_MODEL_KWARGS

    model = SentenceTransformer(model_name, device=device, truncate_dim=768, **extra_kwargs)

    tok = getattr(model, "tokenizer", None)
    if tok is None:
        first = model._first_module()
        tok = getattr(first, "tokenizer", None)
    if tok is None:
        raise RuntimeError("Could not find tokenizer on SentenceTransformer model")

    return model, tok, device


# ============================================================
# Token counting + embedding
# ============================================================

def count_tokens_batch(tokenizer: Any, texts: list[str]) -> int:
    """
    Count tokens for a batch of texts using the model tokenizer.
    Uses add_special_tokens=False to track "content tokens" only.
    """
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return sum(len(ids) for ids in enc["input_ids"])


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: Optional[int] = None,
    normalize_embeddings: Optional[bool] = None,
    convert_to_numpy: bool = True,
) -> np.ndarray:
    """
    Encode texts into embeddings as a float32 numpy array.

    Args:
      batch_size: overrides BATCH_SIZE if provided
      normalize_embeddings: overrides NORMALIZE_EMBEDDINGS if provided
    """
    batch_size = batch_size if batch_size is not None else BATCH_SIZE
    normalize_embeddings = normalize_embeddings if normalize_embeddings is not None else NORMALIZE_EMBEDDINGS

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=convert_to_numpy,
        normalize_embeddings=normalize_embeddings,
    )
    if not isinstance(emb, np.ndarray):
        emb = np.asarray(emb)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)
    return emb
