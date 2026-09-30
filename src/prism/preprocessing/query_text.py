"""Lightweight deterministic normalization that preserves code terminology."""

from __future__ import annotations

import re

_HORIZONTAL_WHITESPACE = re.compile(r"[\t\f\v ]+")
_EXCESS_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def normalize_query(query: str) -> str:
    """Normalize line endings and repeated whitespace without rewriting tokens."""
    text = str(query).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()
