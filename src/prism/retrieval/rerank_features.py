"""Deterministic, unfitted features for dense/BM25 candidate diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from prism.preprocessing.code_tokens import tokenize_programming_text

_IDENTIFIER = re.compile(r"(?u)\b[A-Za-z_][A-Za-z_0-9]*\b")
FEATURE_NAMES = (
    "dense_cosine_similarity",
    "bm25_score",
    "dense_rank",
    "bm25_rank",
    "dense_reciprocal_rank",
    "bm25_reciprocal_rank",
    "query_token_count",
    "candidate_token_count",
    "lexical_overlap_count",
    "lexical_overlap_fraction",
    "identifier_overlap_count",
    "identifier_overlap_fraction",
    "exact_phrase_bigram_count",
    "query_terms_in_candidate",
    "query_term_fraction",
)


def identifier_set(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _IDENTIFIER.finditer(text)}


def extract_pair_features(
    query: str,
    candidate: str,
    *,
    dense_score: float,
    bm25_score: float,
    dense_rank: int | None,
    bm25_rank: int | None,
    query_tokens: Sequence[str] | None = None,
    candidate_tokens: Sequence[str] | None = None,
    query_identifiers: set[str] | None = None,
    candidate_identifiers: set[str] | None = None,
    candidate_bigrams: set[tuple[str, str]] | None = None,
) -> dict[str, float | int]:
    """Extract the requested lexical, identifier, phrase, and rank features."""
    q_tokens = list(query_tokens) if query_tokens is not None else tokenize_programming_text(query)
    d_tokens = list(candidate_tokens) if candidate_tokens is not None else tokenize_programming_text(candidate)
    q_set, d_set = set(q_tokens), set(d_tokens)
    matched = q_set & d_set
    q_identifiers = query_identifiers if query_identifiers is not None else identifier_set(query)
    d_identifiers = candidate_identifiers if candidate_identifiers is not None else identifier_set(candidate)
    identifier_matches = q_identifiers & d_identifiers
    q_bigrams = list(zip(q_tokens, q_tokens[1:]))
    d_bigrams = candidate_bigrams if candidate_bigrams is not None else set(zip(d_tokens, d_tokens[1:]))
    exact_bigram_matches = sum(pair in d_bigrams for pair in q_bigrams)
    query_term_count = len(q_set)

    return {
        "dense_cosine_similarity": float(dense_score),
        "bm25_score": float(bm25_score),
        "dense_rank": int(dense_rank or 0),
        "bm25_rank": int(bm25_rank or 0),
        "dense_reciprocal_rank": 1.0 / dense_rank if dense_rank else 0.0,
        "bm25_reciprocal_rank": 1.0 / bm25_rank if bm25_rank else 0.0,
        "query_token_count": len(q_tokens),
        "candidate_token_count": len(d_tokens),
        "lexical_overlap_count": len(matched),
        "lexical_overlap_fraction": len(matched) / query_term_count if query_term_count else 0.0,
        "identifier_overlap_count": len(identifier_matches),
        "identifier_overlap_fraction": (
            len(identifier_matches) / len(q_identifiers) if q_identifiers else 0.0
        ),
        "exact_phrase_bigram_count": exact_bigram_matches,
        "query_terms_in_candidate": len(matched),
        "query_term_fraction": len(matched) / query_term_count if query_term_count else 0.0,
    }


def diagnostic_scores(
    rows: Sequence[tuple[str, Mapping[str, float | int]]],
) -> dict[str, float]:
    """Make a fixed, per-query heuristic score; this function fits no weights.

    Dense/BM25 raw scores are min-max scaled within the candidate union. Rank
    signals and overlap fractions are already bounded. Seven signals receive
    equal weight; labels are never read here.
    """
    if not rows:
        return {}

    def minmax(key: str) -> dict[str, float]:
        values = [float(features[key]) for _, features in rows]
        low, high = min(values), max(values)
        if high <= low:
            return {doc_id: 0.0 for doc_id, _ in rows}
        return {
            doc_id: (float(features[key]) - low) / (high - low)
            for doc_id, features in rows
        }

    dense_scaled = minmax("dense_cosine_similarity")
    bm25_scaled = minmax("bm25_score")
    scores: dict[str, float] = {}
    for doc_id, features in rows:
        score = (
            dense_scaled[doc_id]
            + bm25_scaled[doc_id]
            + float(features["dense_reciprocal_rank"])
            + float(features["bm25_reciprocal_rank"])
            + float(features["lexical_overlap_fraction"])
            + float(features["identifier_overlap_fraction"])
            + float(features["exact_phrase_bigram_count"])
            / max(1, int(features["query_token_count"]) - 1)
        ) / 7.0
        scores[doc_id] = score
    return scores
