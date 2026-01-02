#!/usr/bin/env python3
import glob
import os
import re
from collections import Counter

DIR = "gnis"

word_freq = Counter()
bigram_freq = Counter()

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

for path in glob.glob(os.path.join(DIR, "*.txt")):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue

            tokens = tokenize(parts[1])
            if not tokens:
                continue

            word_freq.update(tokens)
            bigram_freq.update(
                (tokens[i], tokens[i + 1])
                for i in range(len(tokens) - 1)
            )

# DROP single-character or infrequent words
word_freq = Counter({w: c for w, c in word_freq.items() if len(w) > 1 and c >= 3})

# write words
with open("gnis-words.txt", "w", encoding="utf-8") as out:
    for word, count in word_freq.items():
        out.write(f"{word} {count}\n")

# write bigrams
with open("gnis-bigrams.txt", "w", encoding="utf-8") as out:
    for (w1, w2), count in bigram_freq.items():
        out.write(f"{w1} {w2} {count}\n")
