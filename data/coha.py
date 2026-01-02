#!/usr/bin/env python3
import glob
import os
import re
from collections import Counter

DIR = "coha-ngrams"   # <-- change this
PREFIX = "coha"        # outputs: decade-words.txt, decade-bigrams.txt

word_freq = Counter()
bigram_freq = Counter()

# keep only letters plus - . '
VALID_WORD = re.compile(r"^[a-z\-\.\']+$")

def letters_len(s: str) -> int:
    return sum(1 for ch in s if "a" <= ch <= "z")

for path in glob.glob(os.path.join(DIR, "*.txt")):
    print(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()  # <freq> <word1> <word2> <decade>
            if len(parts) < 4:
                continue

            try:
                freq = int(parts[0])
            except ValueError:
                continue

            w1 = parts[1].lower()
            w2 = parts[2].lower()

            # discard whole bigram if either word has disallowed characters
            if not (VALID_WORD.match(w1) and VALID_WORD.match(w2)):
                continue

            # discard bigram if total LETTER characters across both words < 4
            if letters_len(w1) + letters_len(w2) < 4:
                continue

            bigram_freq[(w1, w2)] += freq

            # words can be part of bigrams even if <4 chars; we just won't SAVE them later
            word_freq[w1] += freq
            word_freq[w2] += freq

# write words (only >= 4 chars)
with open(f"{PREFIX}-words.txt", "w", encoding="utf-8") as out:
    for word, count in word_freq.items():
        if len(word) >= 4 and count > 10:
            out.write(f"{word} {count}\n")

# write bigrams
with open(f"{PREFIX}-bigrams.txt", "w", encoding="utf-8") as out:
    for (w1, w2), count in bigram_freq.items():
        out.write(f"{w1} {w2} {count}\n")
