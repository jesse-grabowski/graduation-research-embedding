# bench_mp.py
# preprocess -> spellcheck -> sentencize -> chunk (mp) -> embed (main process) -> write Parquet
# Processes YEARS sequentially from years.txt (one year per line).
# Reports ONLY tokens/sec (wall-clock) per-year.
#
# SQLite tracking:
# - embedded_articles(year, article_id) marks an article done
# - embedded_years(year) marks a full year done
#
# We mark an article "done" only when the *last kept chunk* for that article
# has been embedded (detected via is_last flag in the embedding buffer).
#
# Parquet output:
# - Writes embeddings to "<YEAR>.parquet" with upsert semantics on (article_id, chunk_id)
# - Columns: article_id, chunk_id, chunk_text, embedding
# - Uses a staging parquet "<YEAR>.staging.parquet" during the run and merges/dedupes at the end.
#
# Crash recovery:
# - If "<YEAR>.staging.parquet" exists at startup, we try to merge it into "<YEAR>.parquet"
#   BEFORE checking year_done / starting new work.
# - If staging is corrupt/unreadable, we rename it to "<YEAR>.staging.corrupt.<ts>.parquet".

from __future__ import annotations

import os
import uuid
import time
import sqlite3
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Iterator, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import embedder_st as embcfg  # embedding model knobs live here

from spellcheck import init_spellchecker, fix_spelling
from sentencizer import build_nlp, spacy_sentences
from preprocess import preprocess
from data import american_stories

# ============================================================
# CONFIGURATION (edit here)
# ============================================================

YEARS_FILE = "years.txt"   # one year per line, comments allowed with '#'

DATASET = american_stories

# Multiprocessing (chunking)
N_PROCESSES = min(cpu_count(), 12)
POOL_CHUNKSIZE = 4

# Dataset
DOC_LIMIT = None        # None = no limit (note: year completion requires full run)
PROGRESS_EVERY = 500    # docs; set to 0 to disable

# Chunking
WINDOW_SENTENCES = 2
STRIDE_SENTENCES = 1
MIN_CHUNK_WORDS = 16

# Parquet output
PARQUET_COMPRESSION = "zstd"  # good speed/size tradeoff

# Resolve project root (src/..)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# NEW: Put ALL sqlite + parquet outputs in project_root/out
OUT_DIR = PROJECT_ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Progress tracking DB (in out/)
SQLITE_PATH = str(OUT_DIR / "embedded_progress.sqlite")

# ============================================================
# Metrics
# ============================================================

@dataclass
class YearMetrics:
    # Parquet
    parquet_stage_write_s: float = 0.0
    parquet_merge_s: float = 0.0
    parquet_stage_batches: int = 0

    # SQLite
    sqlite_read_s: float = 0.0
    sqlite_write_s: float = 0.0
    sqlite_commits: int = 0
    sqlite_reads: int = 0
    sqlite_writes: int = 0

    def total_parquet_s(self) -> float:
        return self.parquet_stage_write_s + self.parquet_merge_s

    def total_sqlite_s(self) -> float:
        return self.sqlite_read_s + self.sqlite_write_s


@contextmanager
def timed(add_fn):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        add_fn(time.perf_counter() - t0)

# ============================================================
# Globals initialized per worker
# ============================================================

_NLP = None
_WINDOW_SENTENCES = WINDOW_SENTENCES
_STRIDE_SENTENCES = STRIDE_SENTENCES


# ============================================================
# Years file
# ============================================================

def read_years(path: str) -> List[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Years file not found: {path}")

    years: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            # allow inline comments
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
            try:
                y = int(line)
                years.append(y)
            except ValueError:
                raise ValueError(f"Invalid year in {path!r}: {raw!r}")

    if not years:
        raise ValueError(f"No years found in {path}")
    return years


# ============================================================
# SQLite helpers (MAIN PROCESS ONLY)
# ============================================================

def init_db(conn: sqlite3.Connection):
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedded_articles (
          year       INTEGER NOT NULL,
          article_id TEXT    NOT NULL,
          PRIMARY KEY (year, article_id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedded_years (
          year INTEGER NOT NULL PRIMARY KEY
        )
        """
    )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_embedded_articles_year ON embedded_articles(year)")
    conn.commit()


def year_done(conn: sqlite3.Connection, year: int, metrics: Optional[YearMetrics] = None) -> bool:
    if metrics is None:
        cur = conn.execute("SELECT 1 FROM embedded_years WHERE year=?", (year,))
        return cur.fetchone() is not None

    with timed(lambda dt: setattr(metrics, "sqlite_read_s", metrics.sqlite_read_s + dt)):
        metrics.sqlite_reads += 1
        cur = conn.execute("SELECT 1 FROM embedded_years WHERE year=?", (year,))
        return cur.fetchone() is not None


def mark_year_done(conn: sqlite3.Connection, year: int, metrics: Optional[YearMetrics] = None):
    if metrics is None:
        conn.execute("INSERT OR IGNORE INTO embedded_years(year) VALUES (?)", (year,))
        conn.commit()
        return

    with timed(lambda dt: setattr(metrics, "sqlite_write_s", metrics.sqlite_write_s + dt)):
        metrics.sqlite_writes += 1
        conn.execute("INSERT OR IGNORE INTO embedded_years(year) VALUES (?)", (year,))
        conn.commit()
        metrics.sqlite_commits += 1


def article_done(conn: sqlite3.Connection, year: int, article_id: str, metrics: Optional[YearMetrics] = None) -> bool:
    if metrics is None:
        cur = conn.execute(
            "SELECT 1 FROM embedded_articles WHERE year=? AND article_id=?",
            (year, article_id),
        )
        return cur.fetchone() is not None

    with timed(lambda dt: setattr(metrics, "sqlite_read_s", metrics.sqlite_read_s + dt)):
        metrics.sqlite_reads += 1
        cur = conn.execute(
            "SELECT 1 FROM embedded_articles WHERE year=? AND article_id=?",
            (year, article_id),
        )
        return cur.fetchone() is not None


def mark_articles_done(
    conn: sqlite3.Connection,
    year: int,
    article_ids: List[str],
    metrics: Optional[YearMetrics] = None,
):
    if not article_ids:
        return
    uniq = list({aid for aid in article_ids if aid is not None})
    if not uniq:
        return

    if metrics is None:
        conn.executemany(
            "INSERT OR IGNORE INTO embedded_articles(year, article_id) VALUES (?, ?)",
            [(year, aid) for aid in uniq],
        )
        conn.commit()
        return

    with timed(lambda dt: setattr(metrics, "sqlite_write_s", metrics.sqlite_write_s + dt)):
        metrics.sqlite_writes += 1
        conn.executemany(
            "INSERT OR IGNORE INTO embedded_articles(year, article_id) VALUES (?, ?)",
            [(year, aid) for aid in uniq],
        )
        conn.commit()
        metrics.sqlite_commits += 1


# ============================================================
# Parquet helpers (MAIN PROCESS ONLY)
# ============================================================

def make_parquet_schema(embedding_dim: int) -> pa.Schema:
    return pa.schema([
        ("article_id", pa.string()),
        ("chunk_id", pa.int32()),
        ("chunk_text", pa.string()),
        ("embedding", pa.list_(pa.float32(), list_size=embedding_dim)),
    ])


def write_batch_to_staging(
    writer: pq.ParquetWriter,
    article_ids: List[str],
    chunk_ids: List[int],
    chunk_texts: List[str],
    emb: np.ndarray,
    embedding_dim: int,
    metrics: Optional[YearMetrics] = None,
) -> None:
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)
    if not emb.flags["C_CONTIGUOUS"]:
        emb = np.ascontiguousarray(emb, dtype=np.float32)

    arr_article = pa.array(article_ids, type=pa.string())
    arr_chunk = pa.array(chunk_ids, type=pa.int32())
    arr_text = pa.array(chunk_texts, type=pa.string())

    flat = pa.array(emb.reshape(-1), type=pa.float32())
    arr_emb = pa.FixedSizeListArray.from_arrays(flat, embedding_dim)

    table = pa.Table.from_arrays(
        [arr_article, arr_chunk, arr_text, arr_emb],
        names=["article_id", "chunk_id", "chunk_text", "embedding"],
    )

    # Time the parquet write itself (includes Arrow->Parquet work in this call)
    if metrics is None:
        writer.write_table(table)
        return

    with timed(lambda dt: setattr(metrics, "parquet_stage_write_s", metrics.parquet_stage_write_s + dt)):
        writer.write_table(table)
    metrics.parquet_stage_batches += 1


def _merge_upsert_parquet_impl(final_path: str, staging_path: str, schema: pa.Schema) -> None:
    staging = pq.read_table(staging_path)

    if os.path.exists(final_path):
        existing = pq.read_table(final_path)
        combined = pa.concat_tables([existing, staging], promote_options="permissive")
    else:
        combined = staging

    rownum = pa.array(np.arange(combined.num_rows, dtype=np.int64))
    combined = combined.append_column("_rownum", rownum)

    if hasattr(combined, "group_by"):
        gb = combined.group_by(["article_id", "chunk_id"]).aggregate([("_rownum", "max")])
        max_col = "_rownum_max" if "_rownum_max" in gb.column_names else gb.column_names[-1]
        indices = gb[max_col].cast(pa.int64())

        winners = combined.take(indices)
        winners = winners.drop(["_rownum"])
        winners = winners.select(schema.names)
    else:
        df = combined.to_pandas()
        df = df.sort_values("_rownum").drop_duplicates(["article_id", "chunk_id"], keep="last")
        df = df.drop(columns=["_rownum"])
        winners = pa.Table.from_pandas(df, preserve_index=False).select(schema.names)

    tmp = f"{final_path}.tmp.{uuid.uuid4().hex}"
    pq.write_table(winners, tmp, compression=PARQUET_COMPRESSION)
    os.replace(tmp, final_path)

    os.remove(staging_path)


def merge_upsert_parquet(final_path: str, staging_path: str, schema: pa.Schema, metrics: Optional[YearMetrics] = None) -> None:
    if not os.path.exists(staging_path):
        return

    if metrics is None:
        _merge_upsert_parquet_impl(final_path, staging_path, schema)
        return

    with timed(lambda dt: setattr(metrics, "parquet_merge_s", metrics.parquet_merge_s + dt)):
        _merge_upsert_parquet_impl(final_path, staging_path, schema)


def recover_staging_if_present(
    year: int,
    parquet_path: str,
    staging_path: str,
    schema: pa.Schema,
    metrics: Optional[YearMetrics] = None,
) -> None:
    if not os.path.exists(staging_path):
        return

    print(f"[{year}] Found existing staging file: {staging_path} -> attempting recovery merge...")
    try:
        merge_upsert_parquet(parquet_path, staging_path, schema, metrics=metrics)
        print(f"[{year}] Recovery merge succeeded.")
    except Exception as e:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrupt_path = str(OUT_DIR / f"{year}.staging.corrupt.{ts}.parquet")
        try:
            os.replace(staging_path, corrupt_path)
        except Exception:
            corrupt_path = staging_path
        print(f"[{year}] Recovery merge failed; staging renamed to: {corrupt_path}")
        print(f"[{year}] Error: {e}")


# ============================================================
# Worker setup
# ============================================================

def init_worker(window_sentences: int, stride_sentences: int):
    global _NLP, _WINDOW_SENTENCES, _STRIDE_SENTENCES
    _WINDOW_SENTENCES = window_sentences
    _STRIDE_SENTENCES = stride_sentences

    init_spellchecker(
        word_dicts=[
            DATA_DIR / "frequency_dictionary_en_82_765.txt",
            DATA_DIR / "gnis-words.txt",
            DATA_DIR / "jrc-names-words.txt",
        ],
        bigram_dicts=[
            DATA_DIR / "frequency_bigramdictionary_en_243_342.txt",
            DATA_DIR / "gnis-bigrams.txt",
            DATA_DIR / "jrc-names-bigrams.txt",
        ],
        max_edit_distance=2,
        prefix_length=7,
    )

    _NLP = build_nlp()


# ============================================================
# Chunking + processing
# ============================================================

def count_words_fast(s: str) -> int:
    if not s:
        return 0
    return s.count(" ") + 1


def make_sentence_windows(sentences: List[str], window: int, stride: int) -> Iterator[str]:
    n = len(sentences)
    if n == 0:
        return

    if n < window:
        yield " ".join(sentences)
        return

    for start in range(0, n - window + 1, stride):
        yield " ".join(sentences[start : start + window])

    last_start = n - window
    if last_start % stride != 0:
        yield " ".join(sentences[last_start : last_start + window])


def process_article(ex: dict):
    global _NLP, _WINDOW_SENTENCES, _STRIDE_SENTENCES

    t0 = time.perf_counter()
    try:
        article_id = ex["article_id"]
        article_text = ex["article"]

        text = preprocess(article_text)
        fixed, _stats = fix_spelling(text)

        sentences = spacy_sentences(fixed, nlp=_NLP)
        if not isinstance(sentences, list):
            sentences = list(sentences)

        chunk_texts = list(make_sentence_windows(sentences, _WINDOW_SENTENCES, _STRIDE_SENTENCES))

        elapsed = time.perf_counter() - t0
        return article_id, chunk_texts, elapsed, True

    except Exception:
        elapsed = time.perf_counter() - t0
        return None, [], elapsed, False


def iter_examples_local(ds, limit: Optional[int] = None) -> Iterator[dict]:
    n = 0
    for ex in ds:
        if limit is not None and n >= limit:
            break
        yield {"article_id": ex["article_id"], "article": ex["article"]}
        n += 1


# ============================================================
# Per-year run
# ============================================================

def run_year(
    year: int,
    conn: sqlite3.Connection,
    embedder,
    tokenizer,
    embedding_dim: int,
    parquet_schema: pa.Schema,
    pool: Pool,
) -> None:
    metrics = YearMetrics()

    # NEW: out/<YEAR>.parquet etc.
    parquet_path = str(OUT_DIR / f"{year}.parquet")
    staging_path = str(OUT_DIR / f"{year}.staging.parquet")

    # Crash recovery per-year (before checking year_done)
    recover_staging_if_present(year, parquet_path, staging_path, parquet_schema, metrics=metrics)

    if year_done(conn, year, metrics=metrics):
        print(f"[{year}] already marked complete in {SQLITE_PATH}. Skipping.")
        return

    # If staging still exists, refuse to proceed (means recovery couldn't resolve)
    if os.path.exists(staging_path):
        raise RuntimeError(
            f"[{year}] staging file still exists after recovery: {staging_path}. "
            "Rename/delete it or inspect the corrupt staging file."
        )

    # Load dataset
    articles = DATASET(year)
    ex_iter = iter_examples_local(articles, limit=DOC_LIMIT)

    staging_writer = pq.ParquetWriter(staging_path, schema=parquet_schema, compression=PARQUET_COMPRESSION)

    # Stats
    n_docs = 0
    n_errors = 0
    total_worker_time = 0.0

    total_tokens_embedded = 0
    n_chunks_embedded = 0
    n_articles_marked_done = 0
    n_articles_skipped = 0

    # Buffer for embedding batches:
    # (article_id, chunk_id, chunk_text, is_last_for_article)
    chunk_buffer: List[Tuple[str, int, str, bool]] = []

    def flush():
        nonlocal chunk_buffer, total_tokens_embedded, n_chunks_embedded, n_articles_marked_done

        if not chunk_buffer:
            return

        texts = [t[2] for t in chunk_buffer]
        total_tokens_embedded += embcfg.count_tokens_batch(tokenizer, texts)

        emb = embcfg.embed_texts(
            embedder,
            texts,
            batch_size=embcfg.BATCH_SIZE,
            normalize_embeddings=embcfg.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )
        if emb.ndim != 2 or emb.shape[1] != embedding_dim:
            raise RuntimeError(f"[{year}] Unexpected embedding shape: {getattr(emb, 'shape', None)}")

        article_ids = [t[0] for t in chunk_buffer]
        chunk_ids = [int(t[1]) for t in chunk_buffer]
        chunk_texts = [t[2] for t in chunk_buffer]

        write_batch_to_staging(
            staging_writer,
            article_ids=article_ids,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            emb=emb,
            embedding_dim=embedding_dim,
            metrics=metrics,
        )

        n_chunks_embedded += len(texts)

        done_article_ids = [aid for (aid, _cid, _txt, is_last) in chunk_buffer if is_last]
        if done_article_ids:
            mark_articles_done(conn, year, done_article_ids, metrics=metrics)
            n_articles_marked_done += len(set(done_article_ids))  # approx

        chunk_buffer = []

    run_start = time.perf_counter()

    # Consume worker results from shared pool
    for article_id, chunk_texts, elapsed, ok in pool.imap_unordered(
        process_article, ex_iter, chunksize=POOL_CHUNKSIZE
    ):
        n_docs += 1
        total_worker_time += elapsed

        if not ok or article_id is None:
            n_errors += 1
            continue

        if article_done(conn, year, article_id, metrics=metrics):
            n_articles_skipped += 1
            continue

        kept: List[Tuple[str, int, str]] = []
        article_chunk_id = 0
        for chunk_text in chunk_texts:
            if MIN_CHUNK_WORDS and count_words_fast(chunk_text) < MIN_CHUNK_WORDS:
                continue
            kept.append((article_id, article_chunk_id, chunk_text))
            article_chunk_id += 1

        if not kept:
            mark_articles_done(conn, year, [article_id], metrics=metrics)
            n_articles_marked_done += 1
            continue

        last_idx = len(kept) - 1
        for i, (aid, cid, txt) in enumerate(kept):
            is_last = (i == last_idx)
            chunk_buffer.append((aid, cid, txt, is_last))

            if len(chunk_buffer) >= embcfg.BATCH_SIZE:
                flush()

        if PROGRESS_EVERY and (n_docs % PROGRESS_EVERY == 0):
            if chunk_buffer:
                flush()

            wall_elapsed = time.perf_counter() - run_start
            toks_per_s = total_tokens_embedded / wall_elapsed if wall_elapsed > 0 else 0.0
            print(
                f"[{year} | {n_docs}] {toks_per_s:,.1f} tokens/sec | "
                f"chunks={n_chunks_embedded:,} | "
                f"done={n_articles_marked_done:,} | skipped={n_articles_skipped:,} | "
                f"errors={n_errors:,}"
            )

    if chunk_buffer:
        flush()

    staging_writer.close()

    # Merge staging into final parquet (upsert)
    merge_upsert_parquet(parquet_path, staging_path, parquet_schema, metrics=metrics)

    # Mark year complete only if full run
    if DOC_LIMIT is None:
        mark_year_done(conn, year, metrics=metrics)
        year_status = "complete (marked)"
    else:
        year_status = "NOT marked (DOC_LIMIT set)"

    wall_clock = time.perf_counter() - run_start
    toks_per_s = total_tokens_embedded / wall_clock if wall_clock > 0 else 0.0
    effective_parallelism = (total_worker_time / wall_clock) if wall_clock > 0 else 0.0

    print(f"\n=== [{year}] Chunking + Embedding + Parquet (tokens/sec) ===")
    print(f"DB: {SQLITE_PATH}")
    print(f"Parquet: {parquet_path}")
    print(f"Year: {year} -> {year_status}")
    print(f"Model: {embcfg.MODEL_NAME} (dim={embedding_dim})")
    print(f"Device: {embcfg.get_device()}")
    print(f"Processes: {N_PROCESSES} | pool chunksize: {POOL_CHUNKSIZE}")
    print(f"Window: {WINDOW_SENTENCES} | Stride: {STRIDE_SENTENCES} | Min chunk words: {MIN_CHUNK_WORDS}")
    print(f"Docs processed: {n_docs:,} (errors: {n_errors:,})")
    print(f"Articles skipped (already done): {n_articles_skipped:,}")
    print(f"Articles marked done (approx):  {n_articles_marked_done:,}")
    print(f"Chunks embedded (this run): {n_chunks_embedded:,}")
    print(f"Total tokens embedded (this run): {total_tokens_embedded:,}")
    print(f"Wall-clock time: {wall_clock:,.3f} s")
    print(f"Throughput: {toks_per_s:,.1f} tokens/sec")
    print(f"Effective parallelism (chunking): {effective_parallelism:,.2f}x\n")

    # IO / tracking metrics
    parquet_s = metrics.total_parquet_s()
    sqlite_s = metrics.total_sqlite_s()

    print("=== IO / Tracking Metrics ===")
    print(f"Parquet staging write time: {metrics.parquet_stage_write_s:,.3f} s "
          f"(batches: {metrics.parquet_stage_batches:,})")
    print(f"Parquet merge/upsert time:  {metrics.parquet_merge_s:,.3f} s")
    print(f"Parquet total time:         {parquet_s:,.3f} s")

    print(f"SQLite read time:           {metrics.sqlite_read_s:,.3f} s (reads: {metrics.sqlite_reads:,})")
    print(f"SQLite write+commit time:   {metrics.sqlite_write_s:,.3f} s "
          f"(writes: {metrics.sqlite_writes:,}, commits: {metrics.sqlite_commits:,})")
    print(f"SQLite total time:          {sqlite_s:,.3f} s")

    if wall_clock > 0:
        print(f"Parquet time share:         {100.0 * parquet_s / wall_clock:,.2f}% of wall-clock")
        print(f"SQLite time share:          {100.0 * sqlite_s / wall_clock:,.2f}% of wall-clock")
    print()


# ============================================================
# Main
# ============================================================

def main():
    years = read_years(YEARS_FILE)

    conn = sqlite3.connect(SQLITE_PATH)
    init_db(conn)

    # Build embedding model once and reuse across all years
    embedder, tokenizer, device = embcfg.build_embedder()
    embedding_dim = int(embedder.get_sentence_embedding_dimension())
    parquet_schema = make_parquet_schema(embedding_dim)

    print(f"Years file: {YEARS_FILE} ({len(years)} years)")
    print(f"Model: {embcfg.MODEL_NAME} (dim={embedding_dim})")
    print(f"Device: {device}")
    print(f"Processes: {N_PROCESSES} | pool chunksize: {POOL_CHUNKSIZE}")
    print(f"Out dir: {OUT_DIR}")
    print()

    # Create ONE pool and reuse across all years (saves spaCy/SymSpell init costs)
    with Pool(
        processes=N_PROCESSES,
        initializer=init_worker,
        initargs=(WINDOW_SENTENCES, STRIDE_SENTENCES),
    ) as pool:
        for year in years:
            run_year(
                year=year,
                conn=conn,
                embedder=embedder,
                tokenizer=tokenizer,
                embedding_dim=embedding_dim,
                parquet_schema=parquet_schema,
                pool=pool,
            )

    conn.close()


if __name__ == "__main__":
    main()
