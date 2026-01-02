import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from symspellpy import SymSpell

_SYM_SPELL: Optional[SymSpell] = None

DEFAULT_MAX_EDIT_DISTANCE = 2

_WS_ONLY_RE = re.compile(r"^\s*$")

# Treat dotted abbreviations as "words" so periods don't float in separators:
# - "N." "Y." "U.S." "N.Y." etc.
WORD_RE = re.compile(
    r"""
    (?:[A-Za-z]\.){2,}                  # U.S. N.Y. etc (2+ dotted letters)
  | (?:[A-Za-z]\.)                      # single-letter abbrev: N. Y.
  | [A-Za-z0-9]+(?:[\'-][A-Za-z0-9]+)*  # normal words with internal ' or -
    """,
    re.VERBOSE,
)

# Conservative cleanup for alignment artifacts (don’t become a general formatter).
# Fixes patterns like: ", .,"  and ", . Williamsburg"
_RE_ARTIFACT_COMMA_DOT_COMMA = re.compile(r",\s*\.\s*,")
_RE_ARTIFACT_COMMA_DOT_WORD = re.compile(r",\s*\.\s+(?=[A-Za-z0-9])")

# Sentence enders we consider "proper"
_END_SENT_RE = re.compile(r"[.!?…]$")
_WEAK_END_RE = re.compile(r"[,;:]$")

# Collapse whitespace + tidy spaces around punctuation/quotes/brackets
_WS_RUN_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{])\s+")
_SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([\)\]\}])")
_SPACE_BEFORE_QUOTES_RE = re.compile(r"\s+([\"'])")
_SPACE_AFTER_QUOTES_RE = re.compile(r"([\"'])\s+")

# --- NEW: extra cleanup rules requested ---
# Ensure a space BEFORE quote marks when they stick to a word (but don't break apostrophes in words)
_SPACE_BEFORE_DQUOTE_STUCK_RE = re.compile(r'(?<=\w)(["“”])')
_SPACE_BEFORE_SQUOTE_STUCK_RE = re.compile(r"(?<=\w)(['’])(?=\W)")  # word' -> word ' ; don't -> don't

# Collapse repeated identical punctuation: ".."->".", "!!"->"!", ",,,"->","
# Does not affect mixed punctuation like ".,"
_COLLAPSE_DUP_PUNCT_RE = re.compile(r"([.!?;,:\u2026])\1+")
# -----------------------------------------


def init_spellchecker(
    *,
    word_dicts: List[str],
    bigram_dicts: List[str],
    max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
    prefix_length: int = 7,
) -> None:
    global _SYM_SPELL

    sym = SymSpell(
        max_dictionary_edit_distance=max_edit_distance,
        prefix_length=prefix_length,
    )

    for path in word_dicts:
        if not sym.load_dictionary(path, term_index=0, count_index=1):
            raise RuntimeError(f"Failed to load word dictionary: {path}")

    for path in bigram_dicts:
        if not sym.load_bigram_dictionary(path, term_index=0, count_index=2):
            raise RuntimeError(f"Failed to load bigram dictionary: {path}")

    _SYM_SPELL = sym


def _require_init() -> SymSpell:
    if _SYM_SPELL is None:
        raise RuntimeError(
            "Spellchecker not initialized. "
            "Call init_spellchecker(word_dicts=[...], bigram_dicts=[...]) first."
        )
    return _SYM_SPELL


@dataclass
class FixStats:
    """
    Uses stdlib difflib.SequenceMatcher:
      similarity = SequenceMatcher(...).ratio()
      normalized_distance = 1 - similarity
    """
    similarity: float = 0.0
    normalized_distance: float = 0.0
    original_length: int = 0
    final_length: int = 0


def _split_preserve_seps(text: str) -> Tuple[List[str], List[str]]:
    """
    Split into words + separators such that:
      len(seps) == len(words) + 1
    and reconstruction is:
      seps[0] + words[0] + seps[1] + ... + words[-1] + seps[-1]
    """
    words: List[str] = []
    seps: List[str] = []
    last = 0

    for m in WORD_RE.finditer(text):
        seps.append(text[last : m.start()])
        words.append(m.group(0))
        last = m.end()

    seps.append(text[last:])
    return words, seps


def _default_joiner_from(left_sep: str, right_sep: str) -> str:
    # Since you don't care about whitespace/newlines, just use a single space.
    return " "


def _last_nonspace_char(s: str) -> str:
    for ch in reversed(s):
        if not ch.isspace():
            return ch
    return ""


def _first_nonspace_char(s: str) -> str:
    for ch in s:
        if not ch.isspace():
            return ch
    return ""


def _merge_leftover_into_trail(leftover: str, trail: str) -> str:
    """
    Combine leftover mids with trail, avoiding duplicated boundary punctuation.
    """
    if not leftover:
        return trail

    ln = _last_nonspace_char(leftover)
    tn = _first_nonspace_char(trail)

    # If the boundary punctuation would repeat, drop the repeated boundary char
    if ln and tn and ln == tn and ln in ",.;:!?":
        i = 0
        while i < len(trail) and trail[i].isspace():
            i += 1
        trail = trail[:i] + trail[i + 1 :]

    return leftover + trail


def _redistribute_seps(
    orig_chunk_seps: List[str],  # length = orig_chunk_words + 1
    new_words: List[str],
    *,
    drop_extra_whitespace_mids: bool = True,
) -> str:
    """
    Rebuild a chunk using original separators as much as possible:
      lead + w0 + mid + w1 + ... + trail

    - Keep lead/trail exactly.
    - Use original mid separators in order for separators between new words.
    - If we need more mids (split), synthesize joiners.
    - If we have leftover mids (merge), append them to trail (with dedupe).
    """
    if not orig_chunk_seps:
        return " ".join(new_words)

    if not new_words:
        return "".join(orig_chunk_seps)

    lead = orig_chunk_seps[0]
    trail = orig_chunk_seps[-1]
    mids = orig_chunk_seps[1:-1]

    out: List[str] = [lead, new_words[0]]

    needed_between = len(new_words) - 1
    used = 0

    for i in range(needed_between):
        if used < len(mids):
            sep = mids[used]
            used += 1
        else:
            sep = _default_joiner_from(lead, trail)
        out.append(sep)
        out.append(new_words[i + 1])

    if used < len(mids):
        leftover_list = mids[used:]
        if drop_extra_whitespace_mids:
            leftover_list = [s for s in leftover_list if not _WS_ONLY_RE.match(s)]
        leftover = "".join(leftover_list)
        trail = _merge_leftover_into_trail(leftover, trail)

    out.append(trail)
    return "".join(out)


def _apply_original_casing_if_same_word(orig: str, new: str) -> str:
    # If word is unchanged except for case, keep original casing.
    return orig if orig.lower() == new.lower() else new


def _transfer_casing_by_alignment(orig_chunk: List[str], new_chunk: List[str]) -> List[str]:
    """
    Transfer casing from orig_chunk -> new_chunk wherever we can match tokens by lowercase
    alignment (handles split/merge/reorder within a small chunk).
    """
    if not orig_chunk or not new_chunk:
        return new_chunk

    a = [w.lower() for w in orig_chunk]
    b = [w.lower() for w in new_chunk]

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = list(new_chunk)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(min(i2 - i1, j2 - j1)):
                out[j1 + k] = orig_chunk[i1 + k]
    return out


def _realign_words_with_seps(
    orig_words: List[str],
    seps: List[str],
    fixed_words: List[str],
) -> str:
    a = [w.lower() for w in orig_words]
    b = [w.lower() for w in fixed_words]

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out_parts: List[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out_parts.append(seps[i1])
            for k in range(i2 - i1):
                o = orig_words[i1 + k]
                f = fixed_words[j1 + k]
                out_parts.append(_apply_original_casing_if_same_word(o, f))
                out_parts.append(seps[i1 + k + 1])

        elif tag == "replace":
            orig_chunk_words = orig_words[i1:i2]
            orig_chunk_seps = seps[i1 : i2 + 1]
            new_chunk_words = fixed_words[j1:j2]

            # preserve original casing wherever words still match (even in split/merge)
            new_chunk_words = _transfer_casing_by_alignment(orig_chunk_words, new_chunk_words)

            out_parts.append(_redistribute_seps(orig_chunk_seps, new_chunk_words))

        elif tag == "delete":
            out_parts.append("".join(seps[i1 : i2 + 1]))

        elif tag == "insert":
            joiner = " "
            ins = fixed_words[j1:j2]
            if ins:
                if out_parts:
                    tail = out_parts[-1]
                    if tail and tail[-1].isalnum():
                        out_parts.append(joiner)
                out_parts.append(joiner.join(ins))

    return "".join(out_parts)


def _cleanup_realign_artifacts(text: str) -> str:
    """
    Fix only common *artifact* introduced by realignment:
      ", .,"  -> ","
      ", . <word>" -> ", <word>"
    """
    text = _RE_ARTIFACT_COMMA_DOT_COMMA.sub(",", text)
    text = _RE_ARTIFACT_COMMA_DOT_WORD.sub(", ", text)
    return text


def _normalize_spacing(text: str) -> str:
    s = text.strip()
    if not s:
        return s

    # 1) Collapse any whitespace run to single spaces
    s = _WS_RUN_RE.sub(" ", s)

    # 2) Remove spaces in the most common wrong places
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)       # "word ," -> "word,"
    s = _SPACE_AFTER_OPEN_RE.sub(r"\1", s)         # "( word" -> "(word"
    s = _SPACE_BEFORE_CLOSE_RE.sub(r"\1", s)       # "word )" -> "word)"
    s = _SPACE_BEFORE_QUOTES_RE.sub(r"\1", s)      # "word '" -> "word'"
    s = _SPACE_AFTER_QUOTES_RE.sub(r"\1", s)       # "' word" -> "'word"

    return s


def _cleanup_quotes_and_punct(text: str) -> str:
    """
    Extra cleanup steps:
      - Ensure a space BEFORE quote characters when stuck to a word (word" -> word ")
        (but do not break apostrophes inside words like don't)
      - Collapse repeated identical punctuation (.. -> ., !!! -> !), but keep mixed like ".,"
    """
    s = text.strip()
    if not s:
        return s

    # Add space before quotes when they're stuck to a word
    s = _SPACE_BEFORE_DQUOTE_STUCK_RE.sub(r" \1", s)
    s = _SPACE_BEFORE_SQUOTE_STUCK_RE.sub(r" \1", s)

    # Collapse repeated identical punctuation marks
    s = _COLLAPSE_DUP_PUNCT_RE.sub(r"\1", s)

    # If quote-spacing introduced double spaces, collapse again
    s = _WS_RUN_RE.sub(" ", s)

    return s


def _ensure_sentence_terminal_punct(text: str, default: str = ".") -> str:
    """
    Ensure the text ends with a proper sentence-ending mark (., !, ?, …).
    Trims whitespace on both ends.
    """
    s = text.strip()
    if not s:
        return s

    if _END_SENT_RE.search(s):
        return s

    if _WEAK_END_RE.search(s):
        return s[:-1] + default

    return s + default


def fix_spelling(
    text: str,
    *,
    max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
) -> Tuple[str, FixStats]:
    sym = _require_init()

    orig_words, seps = _split_preserve_seps(text)

    if not orig_words:
        final_text = _ensure_sentence_terminal_punct(text)
        final_text = _normalize_spacing(final_text)
        final_text = _cleanup_quotes_and_punct(final_text)

        ratio = difflib.SequenceMatcher(None, text, final_text).ratio()
        stats = FixStats(
            similarity=ratio,
            normalized_distance=1.0 - ratio,
            original_length=len(text),
            final_length=len(final_text),
        )
        return final_text, stats

    # Run SymSpell on words-only to avoid punctuation movement
    words_only = " ".join(orig_words)
    suggestions = sym.lookup_compound(
        words_only,
        max_edit_distance=max_edit_distance,
        transfer_casing=True,
        ignore_non_words=True,
    )
    fixed_words_only = suggestions[0].term if suggestions else words_only
    fixed_words = fixed_words_only.split()

    final_text = _realign_words_with_seps(orig_words, seps, fixed_words)
    final_text = _cleanup_realign_artifacts(final_text)
    final_text = _normalize_spacing(final_text)
    final_text = _cleanup_quotes_and_punct(final_text)
    final_text = _ensure_sentence_terminal_punct(final_text)

    ratio = difflib.SequenceMatcher(None, text, final_text).ratio()
    stats = FixStats(
        similarity=ratio,
        normalized_distance=1.0 - ratio,
        original_length=len(text),
        final_length=len(final_text),
    )
    return final_text, stats
