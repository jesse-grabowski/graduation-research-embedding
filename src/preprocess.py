from unidecode import unidecode
import re

# Precompile regexes / constants
WS_RE = re.compile(r"\s+")

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
            parts[-1] = parts[-1][:-1]
            prev_hyphen = True
        else:
            prev_hyphen = False

    return WS_RE.sub(" ", "".join(parts)).strip()

def preprocess(text: str) -> str:
    text = unidecode(text)
    text = collapse_whitespace(text)
    return text