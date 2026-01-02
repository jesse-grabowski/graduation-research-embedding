# parquet_viewer.py
# Query a <YEAR>.parquet file and return top-N semantic matches.
# Uses shared SentenceTransformers component from embedder_st.py.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import embedder_st as embcfg


# ============================================================
# CONFIG (edit here)
# ============================================================

YEAR = 1770
PARQUET_PATH = f"{YEAR}.parquet"

TOP_N = 10
QUERY_TEXT = "sailing, boats"

# How many rows to scan per Parquet batch (memory/speed tradeoff)
BATCH_ROWS = 50_000

# If True, print results at the bottom when run as script
PRINT_RESULTS = True


# ============================================================
# Helpers
# ============================================================

def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _embeddings_to_2d_float32(col: pa.Array) -> np.ndarray:
    """
    Convert an Arrow list/fixed_size_list column of float32 into a (N, D) float32 numpy array.
    This uses to_pylist() for broad pyarrow compatibility (slower than zero-copy).
    """
    arr = np.asarray(col.to_pylist(), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D (N,D), got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class Match:
    score: float
    article_id: str
    chunk_id: int
    chunk_text: str


# ============================================================
# Viewer
# ============================================================

class YearParquetViewer:
    """
    Streams through a year Parquet file and returns top-N semantic matches for a query.

    Expected Parquet schema columns:
      - article_id: string
      - chunk_id: int32/int64
      - chunk_text: string
      - embedding: fixed-size list / list(float32) of size D

    Notes:
      - We stream in batches to avoid loading the full embeddings matrix.
      - If embeddings and query are normalized, cosine similarity = dot product.
    """

    def __init__(
        self,
        year: int,
        parquet_path: Optional[str] = None,
        batch_rows: int = 50_000,
    ):
        self.year = year
        self.parquet_path = parquet_path or f"{year}.parquet"
        self.batch_rows = int(batch_rows)

        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        # Shared embedder component
        self.model, self.tokenizer, self.device = embcfg.build_embedder()
        self.normalize_embeddings = embcfg.NORMALIZE_EMBEDDINGS

        # Read embedding dimension from schema (best-effort)
        pf = pq.ParquetFile(self.parquet_path)
        schema = pf.schema_arrow
        emb_field = schema.field("embedding")
        self.embedding_dim = self._infer_embedding_dim(emb_field.type)

    @staticmethod
    def _infer_embedding_dim(emb_type: pa.DataType) -> Optional[int]:
        """
        Infer embedding dimension from Arrow type where possible.
        Handles:
          - fixed_size_list<item: float>[D]
          - list<item: float> with list_size metadata (pyarrow list_ with list_size)
        Returns None if unknown.
        """
        if pa.types.is_fixed_size_list(emb_type):
            return emb_type.list_size

        # pyarrow list_ with list_size is still ListType, but carries size in repr for some versions.
        # No stable public API to retrieve list_size across all versions, so return None here.
        return None

    def embed_query(self, query: str) -> np.ndarray:
        q = embcfg.embed_texts(
            self.model,
            [query],
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        if q.ndim != 2 or q.shape[0] != 1:
            raise RuntimeError(f"Unexpected query embedding shape: {q.shape}")
        if self.embedding_dim is not None and q.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"Query embedding dim {q.shape[1]} != parquet embedding dim {self.embedding_dim}"
            )
        return q[0].astype(np.float32, copy=False)

    def search(self, query: str, top_n: int = 10) -> List[Match]:
        top_n = int(top_n)
        if top_n <= 0:
            return []

        qv = self.embed_query(query)
        if not self.normalize_embeddings:
            qv = _l2_normalize(qv.reshape(1, -1))[0]

        # Maintain top-k as "min-heap-ish" arrays sorted ascending by score
        top_scores = np.empty((0,), dtype=np.float32)
        top_rows: List[Tuple[str, int, str]] = []

        pf = pq.ParquetFile(self.parquet_path)
        cols = ["article_id", "chunk_id", "chunk_text", "embedding"]

        for batch in pf.iter_batches(batch_size=self.batch_rows, columns=cols):
            article_ids = batch.column(0).to_pylist()
            chunk_ids = batch.column(1).to_pylist()
            chunk_texts = batch.column(2).to_pylist()

            emb = _embeddings_to_2d_float32(batch.column(3))
            if emb.ndim != 2:
                raise RuntimeError(f"Bad embedding batch shape: {emb.shape}")
            if emb.shape[1] != qv.shape[0]:
                raise RuntimeError(
                    f"Embedding dim mismatch: batch {emb.shape[1]} vs query {qv.shape[0]}"
                )

            if not self.normalize_embeddings:
                emb = _l2_normalize(emb, axis=1)

            scores = emb @ qv  # cosine if normalized

            # Update top-k (simple but reliable; fast enough for small top_n)
            for s, aid, cid, txt in zip(scores, article_ids, chunk_ids, chunk_texts):
                s = float(s)
                cid = int(cid)

                if len(top_rows) < top_n:
                    top_rows.append((aid, cid, txt))
                    top_scores = np.append(top_scores, np.float32(s))
                    order = np.argsort(top_scores)
                    top_scores = top_scores[order]
                    top_rows = [top_rows[i] for i in order]
                else:
                    if s > float(top_scores[0]):
                        top_scores[0] = np.float32(s)
                        top_rows[0] = (aid, cid, txt)
                        order = np.argsort(top_scores)
                        top_scores = top_scores[order]
                        top_rows = [top_rows[i] for i in order]

        # Convert to Match list best-first
        matches: List[Match] = []
        for score, (aid, cid, txt) in zip(top_scores[::-1], top_rows[::-1]):
            matches.append(Match(score=float(score), article_id=aid, chunk_id=cid, chunk_text=txt))
        return matches

    @staticmethod
    def format_match(m: Match, max_chars: int = 1000) -> str:
        preview = " ".join(m.chunk_text.split())
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "…"
        return f"score={m.score:.4f} | article_id={m.article_id} | chunk_id={m.chunk_id}\n  {preview}"


# ============================================================
# Script entry
# ============================================================

def main():
    viewer = YearParquetViewer(YEAR, parquet_path=PARQUET_PATH, batch_rows=BATCH_ROWS)
    matches = viewer.search(QUERY_TEXT, top_n=TOP_N)

    print(f"File: {viewer.parquet_path}")
    print(f"Device: {viewer.device}")
    print(f"Model: {embcfg.MODEL_NAME}")
    print(f"Normalize embeddings: {viewer.normalize_embeddings}")
    if viewer.embedding_dim is not None:
        print(f"Embedding dim: {viewer.embedding_dim}")
    print(f"Query: {QUERY_TEXT!r}")
    print()

    if not matches:
        print("No matches.")
        return

    print(f"Top {len(matches)} matches:\n")
    for i, m in enumerate(matches, start=1):
        print(f"{i:2d}. {YearParquetViewer.format_match(m)}\n")


if __name__ == "__main__":
    if PRINT_RESULTS:
        main()
