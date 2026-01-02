#!/usr/bin/env python3
import re
from collections import Counter

INFILE = "jrc-names/entities"

word_freq = Counter()
bigram_freq = Counter()

def tokenize_plus_field(s):
    # split on '+', lowercase, keep letters/numbers only
    tokens = []
    for part in s.split("+"):
        tokens.extend(re.findall(r"[a-z0-9]+", part.lower()))
    return tokens

with open(INFILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        fields = line.split()  # arbitrary whitespace
        if len(fields) < 4:
            continue

        lang = fields[2]
        if lang not in ("u", "en"):
            continue

        tokens = tokenize_plus_field(fields[3])
        if not tokens:
            continue

        word_freq.update(tokens)
        bigram_freq.update((tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1))

# DROP single-character words
word_freq = Counter({w: c for w, c in word_freq.items() if len(w) > 1 and c >= 3})

with open("jrc-names-words.txt", "w", encoding="utf-8") as out:
    for word, count in word_freq.items():
        out.write(f"{word} {count}\n")

with open("jrc-names-bigrams.txt", "w", encoding="utf-8") as out:
    for (w1, w2), count in bigram_freq.items():
        out.write(f"{w1} {w2} {count}\n")
