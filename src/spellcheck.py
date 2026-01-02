import re
from typing import Iterable, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass

from symspellpy import SymSpell, Verbosity

# -----------------------------------------------------------------------------
# PRODUCTION INITIALIZER
# -----------------------------------------------------------------------------
#
# Goal: avoid import-time heavy work and allow clean per-process initialization
# (e.g., multiprocessing, Ray workers). Behavior should match your current code.
#
# Usage:
#   init_spellchecker(word_dicts=[...], bigram_dicts=[...])
#   fixed, stats = fix_spelling(text)
#
# If you forget to call init_spellchecker(), fix_spelling() raises clearly.
# -----------------------------------------------------------------------------

# Globals set by init_spellchecker()
_SYM_SPELL: Optional[SymSpell] = None

# Precompiled regexes / constants (safe, small)
TOKEN_RE = re.compile(r"[A-Za-z0-9'-]+|[^\w\s]+|\s+")
HAS_WORD_CHARS = re.compile(r"[A-Za-z0-9]")
TOKEN_RE2 = re.compile(r"[A-Za-z0-9'-]+|\s+|[^\w\s]+")
PROPER_NOUN_RE = re.compile(r"^[A-Z]")

LEADING_WS_RE = re.compile(r"^\s+")
TRAILING_WS_RE = re.compile(r"\s+$")
LONG_SQUISHED_RE = re.compile(r"[A-Za-z]{10,}")
WORD_TOKEN_FULL_RE = re.compile(r"[A-Za-z0-9'-]+")

# Default substitution rules (you can override in init_spellchecker if desired)
DEFAULT_SUB_RULES: List[Tuple[str, str]] = [
    ("rn", "m"),
    ("m", "rn"),
    ("vv", "w"),
    ("f", "s"),
    ("f", "c"),
    ("ty", "tly"),
    ("faf", "fast"),
    ("tt", "st"),
    ("]", "i"),
]

# Cached config set by init
_SUB_RULES: List[Tuple[str, str]] = DEFAULT_SUB_RULES
_MAX_EDIT_DISTANCE: int = 2
_MAX_SUBSTITUTIONS: int = 2
_PREFIX_LENGTH: int = 7


def init_spellchecker(
    *,
    word_dicts: List[str],
    bigram_dicts: List[str],
    max_edit_distance: int = 2,
    prefix_length: int = 7,
    sub_rules: Optional[List[Tuple[str, str]]] = None,
    max_substitutions: int = 2,
) -> None:
    """
    Initialize the global SymSpell instance and configuration.
    Call once per process (e.g., inside a worker initializer).
    """
    global _SYM_SPELL, _SUB_RULES, _MAX_EDIT_DISTANCE, _MAX_SUBSTITUTIONS, _PREFIX_LENGTH

    _MAX_EDIT_DISTANCE = max_edit_distance
    _MAX_SUBSTITUTIONS = max_substitutions
    _PREFIX_LENGTH = prefix_length
    _SUB_RULES = sub_rules if sub_rules is not None else DEFAULT_SUB_RULES

    sym_spell = SymSpell(
        max_dictionary_edit_distance=max_edit_distance,
        prefix_length=prefix_length
    )

    for path in word_dicts:
        if not sym_spell.load_dictionary(path, term_index=0, count_index=1):
            raise RuntimeError(f"Failed to load word dictionary: {path}")

    for path in bigram_dicts:
        if not sym_spell.load_bigram_dictionary(path, term_index=0, count_index=2):
            raise RuntimeError(f"Failed to load bigram dictionary: {path}")

    _SYM_SPELL = sym_spell


def _require_init() -> SymSpell:
    if _SYM_SPELL is None:
        raise RuntimeError(
            "Spellchecker not initialized. Call init_spellchecker(word_dicts=[...], bigram_dicts=[...]) first."
        )
    return _SYM_SPELL


# -----------------------------------------------------------------------------
# Dictionary membership
# -----------------------------------------------------------------------------

def in_dictionary(term: str) -> bool:
    sym = _require_init()
    return term in sym.words


# -----------------------------------------------------------------------------
# Variant generation (BFS, bounded) - optimized bindings
# -----------------------------------------------------------------------------

def generate_substitution_variants(
    word: str,
    rules: List[Tuple[str, str]],
    max_subs: int = 2
) -> Iterable[str]:
    """
    Generate variants with up to max_subs substitutions using BFS.
    Behavior matches your current implementation.

    Micro-opts:
      - local bindings for speed
      - avoid repeated attribute lookups
    """
    visited = {word}
    q = deque([(word, 0)])
    visited_add = visited.add
    q_append = q.append
    q_popleft = q.popleft

    while q:
        current, depth = q_popleft()
        if depth >= max_subs:
            continue

        for src, dst in rules:
            start = 0
            find = current.find
            src_len = len(src)
            while True:
                idx = find(src, start)
                if idx == -1:
                    break

                cand = current[:idx] + dst + current[idx + src_len:]
                start = idx + 1

                if cand in visited:
                    continue
                visited_add(cand)

                yield cand
                q_append((cand, depth + 1))


# -----------------------------------------------------------------------------
# Word fixing
# -----------------------------------------------------------------------------

def _restore_case(original: str, fixed: str) -> str:
    if original.isupper():
        return fixed.upper()
    if original.istitle():
        return fixed.title()
    return fixed


def fix_word(
    word: str,
    *,
    use_symspell_fallback: bool = True,
    max_edit_distance: Optional[int] = None,
    max_substitutions: Optional[int] = None
) -> str:
    """
    1) If word is already in dictionary -> keep
    2) Try substitutions (first dictionary match wins)
    3) Optional SymSpell fallback
    """
    sym = _require_init()
    med = _MAX_EDIT_DISTANCE if max_edit_distance is None else max_edit_distance
    msubs = _MAX_SUBSTITUTIONS if max_substitutions is None else max_substitutions

    lower = word.lower()

    # Fast dict membership
    if lower in sym.words:
        return word

    # Substitution BFS
    rules = _SUB_RULES
    for cand in generate_substitution_variants(lower, rules, max_subs=msubs):
        if cand in sym.words:
            return _restore_case(word, cand)

    # SymSpell fallback
    if use_symspell_fallback:
        suggestions = sym.lookup(lower, Verbosity.TOP, max_edit_distance=med)
        if suggestions:
            return _restore_case(word, suggestions[0].term)

    return word


# -----------------------------------------------------------------------------
# Stats (Option A: final denom uses final word count)
# -----------------------------------------------------------------------------

@dataclass
class FixStats:
    original_words: int = 0
    changed_words: int = 0
    final_words: int = 0
    final_in_dict_words: int = 0

    @property
    def percent_changed(self) -> float:
        return 0.0 if self.original_words == 0 else (self.changed_words / self.original_words)

    @property
    def final_in_dict_fraction(self) -> float:
        return 0.0 if self.final_words == 0 else (self.final_in_dict_words / self.final_words)


# -----------------------------------------------------------------------------
# First pass: word-by-word + stats
# -----------------------------------------------------------------------------

def fix_text_with_stats(text: str, *, use_symspell_fallback: bool = True):
    sym = _require_init()
    out: List[str] = []
    out_append = out.append

    stats = FixStats()

    for token in TOKEN_RE.findall(text):
        if HAS_WORD_CHARS.search(token):
            stats.original_words += 1
            fixed = fix_word(token, use_symspell_fallback=use_symspell_fallback)
            if fixed != token:
                stats.changed_words += 1
            out_append(fixed)
        else:
            out_append(token)

    return "".join(out), stats


# -----------------------------------------------------------------------------
# SymSpell compound pass (built-in bigram logic)
# Preserves punctuation by only compounding word/space spans.
# Avoids splitting across proper nouns by treating Capitalized tokens as boundaries.
# -----------------------------------------------------------------------------

def _is_proper_noun_token(tok: str) -> bool:
    # Faster than tok[0].isupper() while preserving behavior for your token set.
    return bool(tok) and bool(PROPER_NOUN_RE.match(tok))


def _compound_span_preserve_ws(span: str, max_edit_distance: int) -> str:
    """
    Run SymSpell lookup_compound() on a word/space span while preserving
    leading/trailing whitespace so punctuation doesn't get glued.
    """
    sym = _require_init()

    m1 = LEADING_WS_RE.match(span)
    m2 = TRAILING_WS_RE.search(span)

    lead = m1.group(0) if m1 else ""
    trail = m2.group(0) if m2 else ""

    core = span[len(lead):]
    if trail:
        core = core[:-len(trail)]

    if not core:
        return span

    suggestions = sym.lookup_compound(core.lower(), max_edit_distance=max_edit_distance)
    fixed_core = suggestions[0].term if suggestions else core

    return lead + fixed_core + trail


def _count_final_words(text: str):
    sym = _require_init()
    total = 0
    in_dict = 0
    for tok in TOKEN_RE.findall(text):
        if HAS_WORD_CHARS.search(tok):
            total += 1
            if tok.lower() in sym.words:
                in_dict += 1
    return total, in_dict


# -----------------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------------

def fix_spelling(
    text: str,
    *,
    use_symspell_fallback: bool = True,
    compound_max_edit_distance: int = 2
):
    """
    Pipeline:
      1) word-by-word pass (substitutions + optional SymSpell lookup)
      2) SymSpell lookup_compound() pass for spacing/squishing (uses bigrams)
         - punctuation preserved by not including it in compound spans
         - proper nouns protected by treating Capitalized tokens as hard boundaries

    Returns (final_text, FixStats)
    """
    _require_init()

    first, stats = fix_text_with_stats(text, use_symspell_fallback=use_symspell_fallback)

    tokens = TOKEN_RE2.findall(first)

    out: List[str] = []
    out_append = out.append
    buf: List[str] = []

    def flush_buf():
        if not buf:
            return
        span = "".join(buf)

        # Heuristic: only run compound when span has spaces or a long squished token
        if (" " in span) or LONG_SQUISHED_RE.search(span):
            out_append(_compound_span_preserve_ws(span, compound_max_edit_distance))
        else:
            out_append(span)

        buf.clear()

    for tok in tokens:
        if WORD_TOKEN_FULL_RE.fullmatch(tok):
            if _is_proper_noun_token(tok):
                flush_buf()
                out_append(tok)
            else:
                buf.append(tok)
        elif tok.isspace():
            buf.append(tok)
        else:
            flush_buf()
            out_append(tok)

    flush_buf()
    final_text = "".join(out)

    stats.final_words, stats.final_in_dict_words = _count_final_words(final_text)
    return final_text, stats


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    init_spellchecker(
        word_dicts=[
            "../data/frequency_dictionary_en_82_765.txt",
            "../data/coha-words.txt",
            "../data/gnis-words.txt",
            "../data/jrc-names-words.txt",
        ],
        bigram_dicts=[
            "../data/frequency_bigramdictionary_en_243_342.txt",
            "../data/coha-bigrams.txt",
            "../data/gnis-bigrams.txt",
            "../data/jrc-names-bigrams.txt",
        ],
        max_edit_distance=2,
        prefix_length=7,
        max_substitutions=2,
        # sub_rules=DEFAULT_SUB_RULES,  # optionally override
    )

    s = (
        "Why, it but this day as was pafing the Temple, "
        "Iickepo came up TO me, and infolenty accofed me with, "
        "Brother, what number 90"
    )

    fixed, stats = fix_spelling(s)

    print(fixed)
    print(f"Percent changed: {stats.percent_changed:.4f}")
    print(f"Final in-dictionary fraction: {stats.final_in_dict_fraction:.4f}")
