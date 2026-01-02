from datasets import load_dataset
from unidecode import unidecode
from spacy.symbols import ORTH
import spacy
from spacy.language import Language
import re

from huggingface_hub import hf_hub_download

def american_stories_local(year: int, cache_dir: str = ".hf_cache"):
    # Download the JSONL file once into a local cache path
    local_path = hf_hub_download(
        repo_id="davidaulloa/AmericanStories",
        filename=f"{year}.jsonl",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    # Now load from local file (non-streaming = fully local reads)
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
    splitlines = text.splitlines
    strip = str.strip
    rstrip = str.rstrip

    joined = ' '.join(
        rstrip(strip(line), '-')
        for line in splitlines()
    )
    return ' '.join(joined.split())



ABBR_WORD = re.compile(r"^[A-Z][A-Za-z]{1,3}$")   # St, Dr, Mr, etc

@Language.component("fix_name_abbrev_sents")
def fix_name_abbrev_sents(doc):
    for i in range(len(doc) - 2):
        w = doc[i]
        dot = doc[i + 1]
        nxt = doc[i + 2]

        # Case: "St" "." "James"  -> don't start a new sentence at "James"
        if ABBR_WORD.match(w.text) and dot.text == ".":
            # Only suppress if next token looks like it continues the phrase
            if nxt.is_alpha or nxt.text in ['"', "'", "”", "’", ")", "]"]:
                nxt.is_sent_start = False

    return doc

nlp = spacy.load("en_core_web_sm")
for abbr in ["inst.", "prox.", "ult."]:
    nlp.tokenizer.add_special_case(abbr, [{ORTH: abbr}])
nlp.add_pipe("fix_name_abbrev_sents", before="parser")

def spacy_sentences(text: str) -> list[str]:
    doc = nlp(text)
    return [s.text for s in doc.sents]

if __name__ == "__main__":
    print(nlp.pipe_names)

    articles = american_stories_local(1770)
    print(articles[0])
    # text = next(iter(articles))["article"]
    text = articles[1]["article"]
    text = unidecode(text)
    text = collapse_whitespace(text)
    print(text)
    sentences = spacy_sentences(text)
    for sentence in sentences:
        print(sentence)