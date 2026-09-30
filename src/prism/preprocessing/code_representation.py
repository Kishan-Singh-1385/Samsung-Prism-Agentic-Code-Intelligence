"""Build a compact, field-aware representation of AppsRetrieval code records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _nonempty_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def starter_is_redundant(starter_code: str, code: str, overlap_threshold: float = 0.8) -> bool:
    """Return true when most distinct starter lines already occur in the solution."""
    starter_lines = set(_nonempty_lines(starter_code))
    if not starter_lines:
        return True
    code_lines = set(_nonempty_lines(code))
    overlap = len(starter_lines & code_lines) / len(starter_lines)
    return overlap >= overlap_threshold


def code_document_representation(
    document: Mapping[str, Any],
    *,
    include_title: bool = True,
    include_language: bool = True,
    include_distinct_starter: bool = True,
    starter_overlap_threshold: float = 0.8,
) -> str:
    """Represent observed title/language/code and only nonredundant starter code.

    URLs and other metadata are deliberately excluded from semantic input.
    Existing solution text is preserved; only outer whitespace is trimmed.
    """
    code = str(document.get("text") or "").strip()
    sections: list[tuple[str, str]] = []

    title = str(document.get("title") or "").strip()
    if include_title and title and title not in code:
        sections.append(("TITLE", title))

    language = str(document.get("language") or "").strip()
    if include_language and language:
        sections.append(("LANGUAGE", language))

    if code:
        sections.append(("CODE", code))

    metadata = document.get("meta_information") or {}
    starter_code = ""
    if isinstance(metadata, Mapping):
        starter_code = str(metadata.get("starter_code") or "").strip()
    if (
        include_distinct_starter
        and starter_code
        and not starter_is_redundant(starter_code, code, starter_overlap_threshold)
    ):
        sections.append(("STARTER_CODE", starter_code))

    return "\n\n".join(f"[{label}]\n{value}" for label, value in sections)
