import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from symspellpy import SymSpell

_SYM_SPELL: Optional[SymSpell] = None

WORD_RE = re.compile(r"[A-Za-z0-9'-]+")

DEFAULT_MAX_EDIT_DISTANCE = 2

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
    original_words: int = 0
    final_words: int = 0
    changed_original_words: int = 0

    @property
    def fraction_changed(self) -> float:
        return 0.0 if self.original_words == 0 else self.changed_original_words / self.original_words


def _word_list(text: str) -> List[str]:
    return WORD_RE.findall(text)

def _count_word_diff(original: str, final: str) -> Tuple[int, int, int]:
    o = [w.lower() for w in _word_list(original)]
    f = [w.lower() for w in _word_list(final)]

    changed = sum(1 for i in range(min(len(o), len(f))) if o[i] != f[i])
    changed += abs(len(o) - len(f))
    return len(o), len(f), changed

def fix_spelling(
    text: str,
    *,
    max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
) -> Tuple[str, FixStats]:
    sym = _require_init()

    suggestions = sym.lookup_compound(
        text,
        max_edit_distance=max_edit_distance,
        transfer_casing=True,
        ignore_non_words=True,
    )

    final_text = suggestions[0].term if suggestions else text

    ow, fw, changed = _count_word_diff(text, final_text)
    stats = FixStats(
        original_words=ow,
        final_words=fw,
        changed_original_words=changed,
    )

    return final_text, stats