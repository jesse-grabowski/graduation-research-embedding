# main.py (your script, updated to use the module)
from datasets import load_dataset
from unidecode import unidecode
from spellcheck import init_spellchecker, fix_spelling
import re

from huggingface_hub import hf_hub_download
from sentencizer import build_nlp, spacy_sentences

# Precompile regexes / constants
WS_RE = re.compile(r"\s+")

def american_stories_local(year: int, cache_dir: str = ".hf_cache"):
    local_path = hf_hub_download(
        repo_id="davidaulloa/AmericanStories",
        filename=f"{year}.jsonl",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    ds = load_dataset("json", data_files=local_path)
    return ds["train"]

def american_stories(year: int):
    ds = load_dataset(
        "davidaulloa/AmericanStories",
        data_files=f"{year}.jsonl",
        streaming=True,
    )
    return ds["train"]

def collapse_whitespace(text: str) -> str:
    parts = []
    prev_hyphen = False
    append = parts.append  # local bind

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if prev_hyphen:
            append(line)
        else:
            append((" " + line) if parts else line)

        if line.endswith("-"):
            # remove trailing hyphen AND prevent a space on next line
            parts[-1] = parts[-1][:-1]
            prev_hyphen = True
        else:
            prev_hyphen = False

    # identical behavior to " ".join("".join(parts).split())
    return WS_RE.sub(" ", "".join(parts)).strip()

if __name__ == "__main__":
    # Initialize SymSpell dictionaries once (required by new spellcheck module)
    init_spellchecker(
        word_dicts=[
            "../data/frequency_dictionary_en_82_765.txt",
            # "../data/coha-words.txt",
            "../data/gnis-words.txt",
            "../data/jrc-names-words.txt",
        ],
        bigram_dicts=[
            "../data/frequency_bigramdictionary_en_243_342.txt",
            # "../data/coha-bigrams.txt",
            "../data/gnis-bigrams.txt",
            "../data/jrc-names-bigrams.txt",
        ],
        max_edit_distance=2,
        prefix_length=7,
        max_substitutions=2,
    )

    # Build spaCy pipeline once
    nlp = build_nlp()

    articles = american_stories_local(1871)

    text = articles[56]["article"]
    text = unidecode(text)
    text = collapse_whitespace(text)

    print(text)

    sentences = spacy_sentences(text, nlp=nlp)

    # printing dominates runtime; keep if you need it
    for sentence in sentences:
        fixed, stats = fix_spelling(sentence)
        print(f"{stats.percent_changed:.2f}, {stats.final_in_dict_fraction:.2f}: ", fixed)
