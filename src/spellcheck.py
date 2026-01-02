import re
import difflib
from typing import Iterable, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass

from symspellpy import SymSpell, Verbosity

# -----------------------------------------------------------------------------
# PRODUCTION INITIALIZER
# -----------------------------------------------------------------------------
#
# Usage:
#   init_spellchecker(word_dicts=[...], bigram_dicts=[...])
#   fixed, stats = fix_spelling(text)
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
    """Generate variants with up to max_subs substitutions using BFS."""
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
# Stats
# -----------------------------------------------------------------------------

@dataclass
class FixStats:
    # Word tokens in the ORIGINAL input text
    original_words: int = 0

    # Stage-1 changes only (word-by-word)
    first_pass_changed_words: int = 0

    # Final token-diff based metrics (includes merges + compound)
    changed_original_words: int = 0   # how many original word tokens were replaced/deleted
    inserted_final_words: int = 0     # how many final word tokens were inserted

    # Final counts
    final_words: int = 0
    final_in_dict_words: int = 0

    @property
    def percent_changed(self) -> float:
        # fraction of ORIGINAL word tokens that changed (replace/delete)
        return 0.0 if self.original_words == 0 else (self.changed_original_words / self.original_words)

    @property
    def final_in_dict_fraction(self) -> float:
        return 0.0 if self.final_words == 0 else (self.final_in_dict_words / self.final_words)


def _word_tokens_lower(text: str) -> List[str]:
    return [t.lower() for t in WORD_TOKEN_FULL_RE.findall(text)]


def _compute_token_diff_stats(original_text: str, final_text: str) -> Tuple[int, int, int, int]:
    """
    Returns:
      original_words, final_words, changed_original_words, inserted_final_words
    """
    orig = _word_tokens_lower(original_text)
    fin = _word_tokens_lower(final_text)

    sm = difflib.SequenceMatcher(a=orig, b=fin)
    changed_orig = 0
    inserted_fin = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            changed_orig += (i2 - i1)
            inserted_fin += (j2 - j1)
        elif tag == "delete":
            changed_orig += (i2 - i1)
        elif tag == "insert":
            inserted_fin += (j2 - j1)

    return len(orig), len(fin), changed_orig, inserted_fin


# -----------------------------------------------------------------------------
# First pass: word-by-word + stats
# -----------------------------------------------------------------------------

def fix_text_with_stats(text: str, *, use_symspell_fallback: bool = True):
    _require_init()
    out: List[str] = []
    out_append = out.append

    stats = FixStats()

    for token in TOKEN_RE.findall(text):
        if HAS_WORD_CHARS.search(token):
            stats.original_words += 1
            fixed = fix_word(token, use_symspell_fallback=use_symspell_fallback)
            if fixed != token:
                stats.first_pass_changed_words += 1
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
# Targeted merge pass for split words (place names etc.)
# -----------------------------------------------------------------------------

def _looks_like_word(tok: str) -> bool:
    return bool(WORD_TOKEN_FULL_RE.fullmatch(tok))


def _merge_case_from_left(left: str, merged_lower: str) -> str:
    # Preserve the left token’s casing style (common for place names)
    if left.isupper():
        return merged_lower.upper()
    if left.istitle():
        return merged_lower.title()
    return merged_lower


def merge_split_words(tokens: List[str], *, max_symspell_edit_distance: int = 0) -> List[str]:
    """
    Merge patterns like:
      - "Eng" + " " + "land" -> "England" (if in dict)
      - "Lancas'" + " " + "ter" -> "Lancaster" (if in dict)

    Only merges if merged candidate is in dictionary (safe).
    Optionally also allow SymSpell TOP with edit distance 0 (default) for dictionary gaps.
    """
    sym = _require_init()
    out: List[str] = []
    i = 0
    n = len(tokens)

    while i < n:
        if (
            i + 2 < n
            and _looks_like_word(tokens[i])
            and tokens[i + 1].isspace()
            and _looks_like_word(tokens[i + 2])
        ):
            w1 = tokens[i]
            ws = tokens[i + 1]
            w2 = tokens[i + 2]

            cand1 = w1 + w2
            cand2 = (w1[:-1] + w2) if w1.endswith("'") else None

            chosen_lower: Optional[str] = None

            c1l = cand1.lower()
            if c1l in sym.words:
                chosen_lower = c1l
            elif cand2 is not None and cand2.lower() in sym.words:
                chosen_lower = cand2.lower()
            else:
                if max_symspell_edit_distance == 0:
                    sug = sym.lookup(c1l, Verbosity.TOP, max_edit_distance=0)
                    if sug and sug[0].term == c1l:
                        chosen_lower = c1l
                    elif cand2 is not None:
                        c2l = cand2.lower()
                        sug2 = sym.lookup(c2l, Verbosity.TOP, max_edit_distance=0)
                        if sug2 and sug2[0].term == c2l:
                            chosen_lower = c2l

            if chosen_lower is not None:
                out.append(_merge_case_from_left(w1, chosen_lower))
                i += 3
                continue

            # no merge: emit w1 + ws and continue from w2
            out.append(w1)
            out.append(ws)
            i += 2
            continue

        out.append(tokens[i])
        i += 1

    return out


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
      2) targeted merge of split tokens: "Eng land" -> "England", "Lancas' ter" -> "Lancaster"
      3) SymSpell lookup_compound() pass for spacing/squishing (uses bigrams)
         - punctuation preserved by not including it in compound spans
         - proper nouns protected by treating Capitalized tokens as hard boundaries

    Returns (final_text, FixStats)
    """
    _require_init()

    first, stats = fix_text_with_stats(text, use_symspell_fallback=use_symspell_fallback)

    tokens = TOKEN_RE2.findall(first)

    # Merge common split-word artifacts safely before compound
    tokens = merge_split_words(tokens, max_symspell_edit_distance=0)

    out: List[str] = []
    out_append = out.append
    buf: List[str] = []

    def flush_buf():
        if not buf:
            return
        span = "".join(buf)

        # only run compound when span has spaces or a long squished token
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

    # Final in-dictionary coverage
    stats.final_words, stats.final_in_dict_words = _count_final_words(final_text)

    # NEW: true change tracking vs original input (includes merges/compound)
    ow, fw, changed_orig, inserted_fw = _compute_token_diff_stats(text, final_text)
    stats.original_words = ow
    stats.final_words = fw
    stats.changed_original_words = changed_orig
    stats.inserted_final_words = inserted_fw

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
        # sub_rules=DEFAULT_SUB_RULES,
    )

    s = (
        "Why, it but this day as was pafing the Temple, "
        "Iickepo came up TO me, and infolenty accofed me with, "
        "Brother, what number 90. "
        "I traveled in New Eng land last year. "
        "We also visited Lancas' ter and had tea."
    )

    fixed, stats = fix_spelling(s)

    print(fixed)
    print(f"Stage-1 changed words: {stats.first_pass_changed_words}")
    print(f"Changed original words (final diff): {stats.changed_original_words}")
    print(f"Inserted final words (final diff): {stats.inserted_final_words}")
    print(f"Percent changed: {stats.percent_changed:.4f}")
    print(f"Final in-dictionary fraction: {stats.final_in_dict_fraction:.4f}")
