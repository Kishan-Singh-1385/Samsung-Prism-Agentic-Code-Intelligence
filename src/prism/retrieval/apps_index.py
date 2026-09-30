"""MiniLM AppsRetrieval index shared by the demo and prediction exporter."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(_ROOT / "artifacts" / "huggingface"))
os.environ.setdefault("MTEB_CACHE", str(_ROOT / "artifacts" / "mteb_cache"))

from prism.retrieval.encoder import SentenceTransformerEncoder


class AppsRetrievalIndex:
    """Load the official AppsRetrieval corpus and score it with baseline MiniLM."""

    def __init__(
        self,
        *,
        model: SentenceTransformerEncoder,
        model_name: str,
        model_revision: str,
        dataset_revision: str,
        corpus_ids: list[str],
        corpus_texts: list[str],
        corpus_embeddings: np.ndarray,
        query_ids: list[str],
        query_texts: list[str],
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.model_revision = model_revision
        self.dataset_revision = dataset_revision
        self.corpus_ids = corpus_ids
        self.corpus_texts = corpus_texts
        self.corpus_embeddings = np.asarray(corpus_embeddings, dtype=np.float32)
        self.query_ids = query_ids
        self.query_texts = query_texts
        self._normalized_corpus = self.corpus_embeddings / np.maximum(
            np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True), 1e-12
        )
        self._descending_id_order = np.empty(len(corpus_ids), dtype=np.int64)
        indices = sorted(range(len(corpus_ids)), key=lambda i: corpus_ids[i], reverse=True)
        self._descending_id_order[indices] = np.arange(len(corpus_ids), dtype=np.int64)

    @classmethod
    def load(cls, config_path: Path, *, device: str | None = None) -> "AppsRetrievalIndex":
        import yaml

        root = _ROOT
        config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        artifact_root = root / "artifacts"
        os.environ.setdefault("HF_HOME", str(artifact_root / "huggingface"))
        os.environ.setdefault("MTEB_CACHE", str(artifact_root / "mteb_cache"))
        import mteb

        task_name = config["evaluation"]["task"]
        dataset_revision = config["evaluation"].get("dataset_revision")
        tasks = mteb.get_tasks(tasks=[task_name])
        if not tasks:
            raise RuntimeError(f"MTEB returned no task for {task_name!r}.")
        task = tasks[0]
        task.load_data()
        subset = next(iter(task.dataset))
        payload = task.dataset[subset]["test"]
        corpus_rows = list(payload["corpus"])
        query_rows = list(payload["queries"])
        corpus_ids = [str(row["id"]) for row in corpus_rows]
        corpus_texts = [
            " ".join(
                str(row[key])
                for key in ("title", "text")
                if row.get(key) not in (None, "")
            )
            for row in corpus_rows
        ]
        query_ids = [str(row["id"]) for row in query_rows]
        query_texts = [str(row.get("text") or "") for row in query_rows]
        dataset_revision = str(
            dataset_revision or task.metadata.dataset.revision
        )

        encoder_config = config["encoder"]
        model_name = str(encoder_config["model_name"])
        model_revision = str(encoder_config["model_revision"])
        cache_key = hashlib.sha256(
            f"{dataset_revision}|{model_name}|{model_revision}|{len(corpus_ids)}|{len(query_ids)}".encode()
        ).hexdigest()[:16]
        cache_filename = f"apps_test_minilm_{cache_key}.npz"
        experiment_cache = artifact_root / "experiments" / "cache" / cache_filename
        demo_cache = artifact_root / "demo" / "cache" / cache_filename

        cached_corpus: np.ndarray | None = None
        cached_queries: np.ndarray | None = None
        cache_path = demo_cache
        for candidate_cache in (experiment_cache, demo_cache):
            metadata_path = candidate_cache.with_suffix(".json")
            if not candidate_cache.exists() or not metadata_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                with np.load(candidate_cache, allow_pickle=False) as values:
                    if (
                        metadata.get("model") == model_name
                        and metadata.get("model_revision") == model_revision
                        and values["corpus_ids"].tolist() == corpus_ids
                    ):
                        cached_corpus = values["corpus_embeddings"].copy()
                        if (
                            values["query_ids"].tolist() == query_ids
                            and values["query_embeddings"].shape[0] == len(query_ids)
                        ):
                            cached_queries = values["query_embeddings"].copy()
                        cache_path = candidate_cache
                        break
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue

        model = SentenceTransformerEncoder(
            model_name,
            device=device or encoder_config.get("device", "cpu"),
            batch_size=int(encoder_config.get("batch_size", 16)),
            revision=model_revision,
        )
        if cached_corpus is None:
            cached_corpus = np.asarray(
                model.encode(
                    corpus_texts,
                    batch_size=int(encoder_config.get("batch_size", 16)),
                    convert_to_numpy=True,
                    show_progress_bar=True,
                ),
                dtype=np.float32,
            )
            cache_path = demo_cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    corpus_ids=np.asarray(corpus_ids),
                    query_ids=np.asarray([], dtype=str),
                    corpus_embeddings=cached_corpus,
                    query_embeddings=np.empty((0, cached_corpus.shape[1]), dtype=np.float32),
                )
            temporary.replace(cache_path)
            cache_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "dataset_revision": dataset_revision,
                        "model": model_name,
                        "model_revision": model_revision,
                        "corpus_count": len(corpus_ids),
                        "query_count": 0,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        index = cls(
            model=model,
            model_name=model_name,
            model_revision=model_revision,
            dataset_revision=dataset_revision,
            corpus_ids=corpus_ids,
            corpus_texts=corpus_texts,
            corpus_embeddings=cached_corpus,
            query_ids=query_ids,
            query_texts=query_texts,
        )
        index._cache_path = cache_path
        index._cache_metadata_path = cache_path.with_suffix(".json")
        index._cached_query_embeddings = cached_queries
        index._batch_size = int(encoder_config.get("batch_size", 16))
        return index

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[dict[str, Any]], float]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        start = time.perf_counter()
        embedding = np.asarray(
            self.model.encode([query], batch_size=self._batch_size, convert_to_numpy=True),
            dtype=np.float32,
        )[0]
        normalized_query = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
        scores = self._normalized_corpus @ normalized_query
        ranked = np.lexsort((self._descending_id_order, -scores))[: min(top_k, len(scores))]
        results = [
            {
                "document_id": self.corpus_ids[int(index)],
                "similarity_score": float(scores[index]),
                "code_preview": " ".join(self.corpus_texts[int(index)].split())[:320],
            }
            for index in ranked
        ]
        return results, time.perf_counter() - start

    def test_query_embeddings(self) -> np.ndarray:
        """Reuse evaluation embeddings, or encode all test queries and cache them."""
        if self._cached_query_embeddings is not None:
            return self._cached_query_embeddings
        embeddings = np.asarray(
            self.model.encode(
                self.query_texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        temporary = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                corpus_ids=np.asarray(self.corpus_ids),
                query_ids=np.asarray(self.query_ids),
                corpus_embeddings=self.corpus_embeddings,
                query_embeddings=embeddings,
            )
        temporary.replace(self._cache_path)
        self._cache_metadata_path.write_text(
            json.dumps(
                {
                    "dataset_revision": self.dataset_revision,
                    "model": self.model_name,
                    "model_revision": self.model_revision,
                    "corpus_count": len(self.corpus_ids),
                    "query_count": len(self.query_ids),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._cached_query_embeddings = embeddings
        return embeddings

    def rank_query_embeddings(
        self, query_embeddings: np.ndarray, top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        normalized_queries = query_embeddings / np.maximum(
            np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12
        )
        output = []
        for query in normalized_queries:
            scores = self._normalized_corpus @ query
            ranked = np.lexsort((self._descending_id_order, -scores))[: min(top_k, len(scores))]
            output.append(
                [(self.corpus_ids[int(i)], float(scores[i])) for i in ranked]
            )
        return output
