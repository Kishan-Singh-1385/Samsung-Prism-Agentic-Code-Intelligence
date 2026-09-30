"""Deterministic lexical tokenization for Python code and problem queries."""

from __future__ import annotations

import re

_WORD_OR_NUMBER = re.compile(r"(?u)(?:[^\W\d]|_)[\w]*|\d+(?:\.\d+)?")
_CAMEL_BOUNDARY_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


def tokenize_programming_text(
    text: str,
    *,
    keep_whole_identifiers: bool = True,
    split_snake_case: bool = True,
    split_camel_case: bool = True,
    keep_numbers: bool = True,
    min_token_length: int = 1,
) -> list[str]:
    """Keep identifiers intact while also exposing their useful components.

    Tokens are case-folded but otherwise preserved. No stopwords, stemming,
    or identifier-frequency filtering is applied.
    """
    tokens: list[str] = []
    for match in _WORD_OR_NUMBER.finditer(str(text)):
        raw = match.group(0)
        if raw[0].isdigit():
            if keep_numbers and len(raw) >= min_token_length:
                tokens.append(raw.casefold())
            continue

        whole_token = raw.casefold()
        if keep_whole_identifiers and len(raw) >= min_token_length:
            tokens.append(whole_token)

        parts = [raw]
        if split_snake_case:
            parts = [part for value in parts for part in value.split("_") if part]
        if split_camel_case:
            split_parts: list[str] = []
            for part in parts:
                part = _CAMEL_BOUNDARY_1.sub(r"\1 \2", part)
                part = _CAMEL_BOUNDARY_2.sub(r"\1 \2", part)
                split_parts.extend(part.split())
            parts = split_parts
        split_changed_identifier = len(parts) != 1 or parts[0].casefold() != whole_token
        if not keep_whole_identifiers or split_changed_identifier:
            tokens.extend(
                part.casefold() for part in parts if len(part) >= min_token_length
            )
    return tokens
