"""Small in-memory BM25 retriever with an optional local pickle cache."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix

from prism.preprocessing.code_tokens import tokenize_programming_text


class BM25Retriever:
    """BM25Okapi over a fixed ordered corpus of document IDs and text."""

    TOKENIZER_VERSION = "programming_tokens_v1"
    INDEX_VERSION = "sparse_postings_v1"

    def __init__(
        self,
        document_ids: list[str],
        tokenized_documents: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        if len(document_ids) != len(tokenized_documents):
            raise ValueError("Document ID and tokenized-document counts differ.")
        if not document_ids:
            raise ValueError("BM25 corpus cannot be empty.")
        self.document_ids = list(document_ids)
        # MTEB breaks equal score ties by document ID descending.
        descending_id_rank = {
            doc_id: rank for rank, doc_id in enumerate(sorted(document_ids, reverse=True))
        }
        self._id_tie_order = np.asarray(
            [descending_id_rank[doc_id] for doc_id in document_ids], dtype=np.int64
        )
        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon = float(epsilon)
        self.tokenizer_options: dict[str, Any] = {}
        self._index = BM25Okapi(
            tokenized_documents, k1=self.k1, b=self.b, epsilon=self.epsilon
        )
        # Store term postings so scoring touches only documents containing a
        # query term instead of scanning the full corpus once per query token.
        self._token_to_column = {term: column for column, term in enumerate(self._index.idf)}
        nnz = sum(len(document) for document in self._index.doc_freqs)
        indices = np.empty(nnz, dtype=np.int32)
        values = np.empty(nnz, dtype=np.float32)
        indptr = np.empty(len(self.document_ids) + 1, dtype=np.int64)
        indptr[0] = 0
        offset = 0
        for row, frequencies in enumerate(self._index.doc_freqs):
            for term, frequency in frequencies.items():
                indices[offset] = self._token_to_column[term]
                values[offset] = frequency
                offset += 1
            indptr[row + 1] = offset
        self._term_postings = csr_matrix(
            (values, indices, indptr),
            shape=(len(self.document_ids), len(self._token_to_column)),
        ).tocsc()
        del indices, values, indptr
        self._document_norm = self.k1 * (
            1.0 - self.b + self.b * np.asarray(self._index.doc_len) / self._index.avgdl
        )

    @classmethod
    def build(
        cls,
        document_ids: list[str],
        documents: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        tokenizer_options: dict[str, Any] | None = None,
    ) -> "BM25Retriever":
        if len(document_ids) != len(documents):
            raise ValueError("Document ID and document-text counts differ.")
        options = tokenizer_options or {}
        tokenized = [tokenize_programming_text(text, **options) for text in documents]
        retriever = cls(
            document_ids, tokenized, k1=k1, b=b, epsilon=epsilon
        )
        retriever.tokenizer_options = dict(options)
        return retriever

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        query_tokens = tokenize_programming_text(query, **self.tokenizer_options)
        if not query_tokens:
            scores = np.zeros(len(self.document_ids), dtype=np.float32)
        else:
            scores = np.zeros(len(self.document_ids), dtype=np.float64)
            query_counts: dict[str, int] = {}
            for token in query_tokens:
                query_counts[token] = query_counts.get(token, 0) + 1
            postings = self._term_postings
            for token, query_frequency in query_counts.items():
                column = self._token_to_column.get(token)
                if column is None:
                    continue
                start, end = postings.indptr[column : column + 2]
                document_indices = postings.indices[start:end]
                term_frequencies = postings.data[start:end]
                idf = self._index.idf.get(token, 0.0)
                scores[document_indices] += (
                    query_frequency
                    * idf
                    * term_frequencies
                    * (self.k1 + 1.0)
                    / (term_frequencies + self._document_norm[document_indices])
                )
        order = np.lexsort((self._id_tie_order, -scores))[: min(top_k, len(scores))]
        return [(self.document_ids[int(i)], float(scores[i])) for i in order]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, cls):
            raise TypeError(f"Unexpected BM25 cache object in {path}")
        return value
