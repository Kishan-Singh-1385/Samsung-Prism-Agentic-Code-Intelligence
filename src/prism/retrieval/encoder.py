"""Small Sentence Transformers encoder compatible with MTEB retrieval tasks."""

from __future__ import annotations

from typing import Any, Mapping

from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoder(SentenceTransformer):
    """A MTEB-compatible SentenceTransformer with legacy retrieval helpers.

    Subclassing the supported SentenceTransformer type lets MTEB apply its
    Sentence Transformers adapter (including task and prompt metadata).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        revision: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_revision = revision
        self.batch_size = batch_size
        super().__init__(model_name, device=device, revision=revision)

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        # MTEB v2 passes task context through these helpers; ST itself consumes
        # the texts and supported encode kwargs.
        for key in ("task_metadata", "hf_split", "hf_subset", "prompt_type"):
            kwargs.pop(key, None)
        return self.encode(queries, batch_size=batch_size or self.batch_size, **kwargs)

    def encode_corpus(
        self,
        corpus: list[Mapping[str, Any]],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Encode corpus records, preserving their observed text fields.

        MTEB retrieval corpora conventionally supply `title` and `text` fields.
        Empty/missing fields are ignored; field values are not normalized or
        otherwise preprocessed.
        """
        texts = []
        for document in corpus:
            parts = [
                str(document[key])
                for key in ("title", "text")
                if document.get(key) not in (None, "")
            ]
            texts.append(" ".join(parts))
        for key in ("task_metadata", "hf_split", "hf_subset", "prompt_type"):
            kwargs.pop(key, None)
        return self.encode(texts, batch_size=batch_size or self.batch_size, **kwargs)
