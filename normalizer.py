#!/usr/bin/env python3
"""Deterministic transcript normalization: fix predictable ASR errors.

Runs AFTER ASR (and after confidence) and BEFORE text reaches OpenCode. No
LLM, no guessing: an ordered list of ``from -> to`` phrase pairs applied with
word boundaries, case-insensitive match, configured ``to`` form kept:

    "open code" -> "OpenCode", "get hub" -> "GitHub"

Conservatism rules (so ordinary English survives):

* multi-word source phrases only in the defaults -- single common words
  ("rust", "code", "windows") are NEVER in the default list;
* whole-phrase match with ``\\b`` boundaries on both ends;
* replacements apply left-to-right, first matching rule wins per span;
* user rules replace the default list wholesale (documented, predictable);
* ``enabled = false`` disables everything (text passes through untouched).
"""

from __future__ import annotations

import re


def compile_rules(pairs: list) -> list:
    """Compile [(from, to)] into [(pattern, to)]. Deterministic order kept."""
    rules = []
    for item in pairs:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            src, dst = item
        elif isinstance(item, dict) and "from" in item and "to" in item:
            src, dst = item["from"], item["to"]
        else:
            continue
        src = str(src).strip()
        dst = str(dst).strip()
        if not src or not dst:
            continue
        # Word boundaries; internal whitespace flexes (ASR spacing varies).
        body = r"\s+".join(re.escape(w) for w in src.split())
        try:
            pat = re.compile(r"\b" + body + r"\b", re.IGNORECASE)
        except re.error:
            continue
        rules.append((pat, dst, src))
    return rules


def normalize(text: str, rules: list) -> tuple[str, int]:
    """Apply rules to text. Returns (new_text, number_of_substitutions)."""
    if not text or not rules:
        return text, 0
    total = 0
    for pat, dst, _src in rules:
        text, n = pat.subn(dst, text)
        total += n
    return text, total


def rules_from_config(cfg: dict) -> list:
    """Build compiled rules; empty list when normalization is disabled."""
    norm = cfg.get("normalization", {})
    if not norm.get("enabled", True):
        return []
    return compile_rules(norm.get("replaces", []))
