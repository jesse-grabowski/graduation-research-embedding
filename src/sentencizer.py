import re
import spacy
from spacy.symbols import ORTH
from spacy.language import Language

# Precompile regexes / constants
ABBR_WORD = re.compile(r"^[A-Z][A-Za-z]{1,3}$")   # St, Dr, Mr, etc
CONT_TOKENS = {'"', "'", "”", "’", ")", "]"}

@Language.component("fix_name_abbrev_sents")
def fix_name_abbrev_sents(doc):
    abbr_match = ABBR_WORD.match
    cont = CONT_TOKENS

    # Case: "St" "." "James"  -> don't start a new sentence at "James"
    for i in range(len(doc) - 2):
        w = doc[i]
        dot = doc[i + 1]
        nxt = doc[i + 2]

        if not abbr_match(w.text):
            continue
        if dot.text != ".":
            continue

        if nxt.is_alpha or nxt.text in cont:
            nxt.is_sent_start = False

    return doc

def build_nlp():
    """
    Build the spaCy pipeline used for sentence splitting.
    Kept identical to your current behavior: uses parser-based sents, with
    a small post-fix component to handle abbreviations.
    """
    # Disable components that don't affect sentence segmentation via parser
    nlp = spacy.load("en_core_web_sm", disable=["ner", "attribute_ruler", "lemmatizer"])

    for abbr in ["inst.", "prox.", "ult."]:
        nlp.tokenizer.add_special_case(abbr, [{ORTH: abbr}])

    nlp.add_pipe("fix_name_abbrev_sents", before="parser")
    return nlp

def spacy_sentences(text: str, nlp=None) -> list[str]:
    """
    Split into sentences with the configured spaCy pipeline.
    If nlp is not provided, builds one (for production, pass a prebuilt nlp).
    """
    if nlp is None:
        nlp = build_nlp()
    doc = nlp(text)
    return [s.text for s in doc.sents]
