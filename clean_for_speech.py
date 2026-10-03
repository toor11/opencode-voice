#!/usr/bin/env python3
"""Strip markdown/code for speech: remove fences, URLs, symbols, truncate."""
import re
import sys

LIMIT = 900


def clean(text: str) -> str:
    text = re.sub(r"```.*?```", " code omitted. ", text, flags=re.S)
    text = re.sub(r"https?://\S+", " link. ", text)
    text = re.sub(r"[#*_`>|~$]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:LIMIT]


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"
    with open(src) as f:
        out = clean(f.read())
    print(out)
    if not out:
        sys.exit(1)
