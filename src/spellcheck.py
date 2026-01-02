import re
from typing import Iterable, List, Tuple
from collections import deque
from dataclasses import dataclass

from symspellpy import SymSpell, Verbosity

# -----------------------------
# SymSpell init + dictionary load
# -----------------------------

max_edit_distance = 2
prefix_length = 7

sym_spell = SymSpell(
    max_dictionary_edit_distance=max_edit_distance,
    prefix_length=prefix_length
)

word_dicts = [
    "../data/frequency_dictionary_en_82_765.txt",
    "../data/coha-words.txt",
    "../data/gnis-words.txt",
    "../data/jrc-names-words.txt",
]

for path in word_dicts:
    if not sym_spell.load_dictionary(path, term_index=0, count_index=1):
        raise RuntimeError(f"Failed to load word dictionary: {path}")

bigram_dicts = [
    "../data/frequency_bigramdictionary_en_243_342.txt",
    "../data/coha-bigrams.txt",
    "../data/gnis-bigrams.txt",
    "../data/jrc-names-bigrams.txt",
]

for path in bigram_dicts:
    if not sym_spell.load_bigram_dictionary(path, term_index=0, count_index=2):
        raise RuntimeError(f"Failed to load bigram dictionary: {path}")

# -----------------------------
# Dictionary membership
# -----------------------------

def in_dictionary(term: str) -> bool:
    return term in sym_spell.words

# -----------------------------
# OCR substitutions (ASCII-only)
# Ordered: earlier = more trusted
# -----------------------------

SUB_RULES: List[Tuple[str, str]] = [
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

# -----------------------------
# Variant generation (BFS, bounded)
# -----------------------------

def generate_substitution_variants(
    word: str,
    rules: List[Tuple[str, str]],
    max_subs: int = 2
) -> Iterable[str]:
    visited = {word}
    q = deque([(word, 0)])

    while q:
        current, depth = q.popleft()
        if depth >= max_subs:
            continue

        for src, dst in rules:
            start = 0
            while True:
                idx = current.find(src, start)
                if idx == -1:
                    break

                cand = current[:idx] + dst + current[idx + len(src):]
                start = idx + 1

                if cand in visited:
                    continue
                visited.add(cand)

                yield cand
                q.append((cand, depth + 1))

# -----------------------------
# Word fixing
# -----------------------------

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
    max_edit_distance: int = 2,
    max_substitutions: int = 2
) -> str:
    lower = word.lower()

    if in_dictionary(lower):
        return word

    for cand in generate_substitution_variants(lower, SUB_RULES, max_subs=max_substitutions):
        if in_dictionary(cand):
            return _restore_case(word, cand)

    if use_symspell_fallback:
        suggestions = sym_spell.lookup(lower, Verbosity.TOP, max_edit_distance=max_edit_distance)
        if suggestions:
            return _restore_case(word, suggestions[0].term)

    return word

# -----------------------------
# Tokenization
# -----------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9'-]+|[^\w\s]+|\s+")
HAS_WORD_CHARS = re.compile(r"[A-Za-z0-9]")

# -----------------------------
# Stats
# -----------------------------

@dataclass
class FixStats:
    original_words: int = 0
    changed_words: int = 0
    final_words: int = 0
    final_in_dict_words: int = 0

    @property
    def percent_changed(self) -> float:
        return 0.0 if self.original_words == 0 else (
            self.changed_words / self.original_words
        )

    @property
    def final_in_dict_fraction(self) -> float:
        return 0.0 if self.final_words == 0 else (
            self.final_in_dict_words / self.final_words
        )

# -----------------------------
# First pass: word-by-word + stats
# -----------------------------

def fix_text_with_stats(text: str, *, use_symspell_fallback: bool = True):
    out = []
    stats = FixStats()

    for token in TOKEN_RE.findall(text):
        if HAS_WORD_CHARS.search(token):
            stats.original_words += 1
            fixed = fix_word(token, use_symspell_fallback=use_symspell_fallback)
            if fixed != token:
                stats.changed_words += 1
            out.append(fixed)
        else:
            out.append(token)

    return "".join(out), stats

# -----------------------------
# Bigram compound pass (safe)
# -----------------------------

TOKEN_RE2 = re.compile(r"[A-Za-z0-9'-]+|\s+|[^\w\s]+")

def _is_proper_noun_token(tok: str) -> bool:
    return bool(tok) and tok[0].isalpha() and tok[0].isupper()

def _compound_span_preserve_ws(span: str, max_edit_distance: int) -> str:
    m1 = re.match(r"^\s+", span)
    m2 = re.search(r"\s+$", span)

    lead = m1.group(0) if m1 else ""
    trail = m2.group(0) if m2 else ""

    core = span[len(lead):]
    if trail:
        core = core[:-len(trail)]

    if not core:
        return span

    suggestions = sym_spell.lookup_compound(core.lower(), max_edit_distance=max_edit_distance)
    fixed_core = suggestions[0].term if suggestions else core

    return lead + fixed_core + trail

def _count_final_words(text: str):
    total = 0
    in_dict = 0
    for tok in TOKEN_RE.findall(text):
        if HAS_WORD_CHARS.search(tok):
            total += 1
            if in_dictionary(tok.lower()):
                in_dict += 1
    return total, in_dict

# -----------------------------
# Full pipeline
# -----------------------------

def fix_spelling(
    text: str,
    *,
    use_symspell_fallback: bool = True,
    compound_max_edit_distance: int = 2
):
    first, stats = fix_text_with_stats(text, use_symspell_fallback=use_symspell_fallback)

    tokens = TOKEN_RE2.findall(first)
    out = []
    buf = []

    def flush_buf():
        if not buf:
            return
        span = "".join(buf)
        if (" " in span) or re.search(r"[A-Za-z]{10,}", span):
            out.append(_compound_span_preserve_ws(span, compound_max_edit_distance))
        else:
            out.append(span)
        buf.clear()

    for tok in tokens:
        if re.fullmatch(r"[A-Za-z0-9'-]+", tok):
            if _is_proper_noun_token(tok):
                flush_buf()
                out.append(tok)
            else:
                buf.append(tok)
        elif tok.isspace():
            buf.append(tok)
        else:
            flush_buf()
            out.append(tok)

    flush_buf()
    final_text = "".join(out)

    stats.final_words, stats.final_in_dict_words = _count_final_words(final_text)
    return final_text, stats

# -----------------------------
# Example
# -----------------------------

if __name__ == "__main__":
    s = (
        "Why, it but this day as was pafing the Temple, "
        "Iickepo came up TO me, and infolenty accofed me with, "
        "Brother, what number 90"
    )

    fixed, stats = fix_spelling(s)

    print(fixed)
    print(f"Percent changed: {stats.percent_changed:.4f}")
    print(f"Final in-dictionary fraction: {stats.final_in_dict_fraction:.4f}")
