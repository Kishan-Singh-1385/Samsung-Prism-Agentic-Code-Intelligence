"""Validate cached Variant B and export its AppsRetrieval test predictions."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts" / "huggingface"))
os.environ.setdefault("MTEB_CACHE", str(ROOT / "artifacts" / "mteb_cache"))

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATASET_REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"
REPRESENTATION_KEY = "python_code_prefix_v1"


def peak_working_set_mib() -> float | None:
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = ctypes.c_void_p
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    if get_info(process(), ctypes.byref(counters), counters.cb):
        return counters.PeakWorkingSetSize / (1024 * 1024)
    return None


def rank_top10(corpus: np.ndarray, queries: np.ndarray, doc_ids: list[str]) -> list[list[tuple[str, float]]]:
    normalized_corpus = corpus / np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-12)
    normalized_queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    desc_id_indices = sorted(range(len(doc_ids)), key=lambda i: doc_ids[i], reverse=True)
    id_tie_order = np.empty(len(doc_ids), dtype=np.int64)
    id_tie_order[desc_id_indices] = np.arange(len(doc_ids), dtype=np.int64)
    rankings: list[list[tuple[str, float]]] = []
    for start in range(0, len(queries), 64):
        scores_block = normalized_queries[start : start + 64] @ normalized_corpus.T
        for scores in scores_block:
            indices = np.lexsort((id_tie_order, -scores))[:10]
            rankings.append([(doc_ids[int(i)], float(scores[i])) for i in indices])
    return rankings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    import mteb
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    started = time.perf_counter()
    task = mteb.get_tasks(tasks=["AppsRetrieval"])[0]
    task.load_data()
    payload = task.dataset[next(iter(task.dataset))]["test"]
    corpus_rows = list(payload["corpus"])
    query_rows = list(payload["queries"])
    doc_ids = [str(row["id"]) for row in corpus_rows]
    query_ids = [str(row["id"]) for row in query_rows]
    qrels = {
        str(qid): {str(doc_id): int(rel) for doc_id, rel in relevant.items()}
        for qid, relevant in payload["relevant_docs"].items()
    }

    cache_sig = hashlib.sha256(
        (DATASET_REVISION + MODEL + MODEL_REVISION + REPRESENTATION_KEY).encode()
    ).hexdigest()[:16]
    cache_path = ROOT / "artifacts" / "experiments" / "cache" / f"cheap_ablation_{cache_sig}.npz"
    metadata_path = cache_path.with_suffix(".json")
    baseline_path = ROOT / "artifacts/experiments/cache/apps_test_minilm_9dc7b770b5cc6fb9.npz"
    corpus_embeddings = None
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "representation": REPRESENTATION_KEY,
            "model": MODEL,
            "revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(f"Variant B cache metadata mismatch: {metadata}")
        with np.load(cache_path, allow_pickle=False) as cached:
            if cached["ids"].tolist() != doc_ids:
                raise RuntimeError("Variant B cache document IDs differ from AppsRetrieval test corpus")
            corpus_embeddings = cached["embeddings"].copy()

    query_embeddings = None
    if baseline_path.is_file():
        with np.load(baseline_path, allow_pickle=False) as cached:
            if cached["corpus_ids"].tolist() != doc_ids or cached["query_ids"].tolist() != query_ids:
                raise RuntimeError("Official baseline cache IDs differ from AppsRetrieval test data")
            query_embeddings = cached["query_embeddings"].copy()

    corpus_cache_hit = corpus_embeddings is not None
    query_cache_hit = query_embeddings is not None

    if corpus_embeddings is None or query_embeddings is None:
        # Fresh checkouts may not include ignored embedding caches. Build only
        # the same frozen-model vectors needed for this candidate validation.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL, device=args.device, revision=MODEL_REVISION)
        if corpus_embeddings is None:
            corpus_texts = [
                " ".join(str(row[key]) for key in ("title", "text") if row.get(key) not in (None, ""))
                for row in corpus_rows
            ]
            representation_texts = [f"python code:\n{text}" for text in corpus_texts]
            corpus_embeddings = np.asarray(model.encode(
                representation_texts, batch_size=args.batch_size,
                convert_to_numpy=True, show_progress_bar=True,
            ), dtype=np.float32)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                np.savez(handle, ids=np.asarray(doc_ids), embeddings=corpus_embeddings)
            metadata_path.write_text(json.dumps({
                "representation": REPRESENTATION_KEY, "model": MODEL,
                "revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION,
            }, indent=2) + "\n", encoding="utf-8")
        if query_embeddings is None:
            raw_queries = [str(row.get("text") or "") for row in query_rows]
            query_embeddings = np.asarray(model.encode(
                raw_queries, batch_size=args.batch_size,
                convert_to_numpy=True, show_progress_bar=True,
            ), dtype=np.float32)
        del model

    retrieval_start = time.perf_counter()
    top10 = rank_top10(corpus_embeddings, query_embeddings, doc_ids)
    predictions: dict[str, dict[str, float]] = {
        qid: {doc_id: score for doc_id, score in rows}
        for qid, rows in zip(query_ids, top10, strict=True)
    }
    metric_result = calculate_retrieval_scores(predictions, qrels, [10])
    ndcg = float(metric_result.ndcg["NDCG@10"])
    mrr = float(metric_result.mrr["MRR@10"])
    retrieval_and_metric_seconds = time.perf_counter() - retrieval_start
    runtime_seconds = time.perf_counter() - started
    peak_mib = peak_working_set_mib()

    candidate_path = ROOT / "artifacts/submission/apps_retrieval_test_top10_variant_b.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format": "mteb-retrieval-results-map-topk-v1",
        "task": "AppsRetrieval",
        "split": "test",
        "dataset_revision": DATASET_REVISION,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "candidate": "Variant B: python code marker",
        "corpus_representation": "'python code:\\n' + official title/text representation (title and text joined by one space)",
        "query_representation": "raw query text",
        "similarity": "cosine",
        "top_k": 10,
        "corpus_size": len(doc_ids),
        "query_count": len(query_ids),
        "predictions": predictions,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(candidate_path)

    summary = {
        "submission": "Samsung PRISM Agentic Code Intelligence",
        "status": "candidate submission system; official baseline preserved",
        "dataset": {
            "task": "CoIR AppsRetrieval", "revision": DATASET_REVISION,
            "available_splits": ["test"], "corpus_documents": len(doc_ids),
            "queries": len(query_ids),
            "relevance_labels_used_only_for_final_metric_calculation": True,
            "test_only_limit": "AppsRetrieval provides only the test split; the measured improvement is exploratory and has not been independently validated.",
        },
        "official_baseline": {
            "model": MODEL, "model_revision": MODEL_REVISION,
            "representation": "official title/text representation; queries raw",
            "ndcg_at_10": 0.06596, "mrr_at_10": 0.05581,
            "source": "previously verified official baseline; baseline artifacts unchanged",
            "prediction_json": "artifacts/submission/apps_retrieval_test_top10.json",
        },
        "final_candidate": {
            "name": "Variant B: python code marker",
            "model": MODEL, "model_revision": MODEL_REVISION,
            "representation": "corpus='python code:\\n'+official title/text; queries raw",
            "ndcg_at_10": ndcg, "mrr_at_10": mrr,
            "ndcg_absolute_change_vs_baseline": ndcg - 0.06596,
            "mrr_absolute_change_vs_baseline": mrr - 0.05581,
            "runtime_seconds": runtime_seconds,
            "retrieval_and_metric_seconds": retrieval_and_metric_seconds,
            "peak_working_set_mib": peak_mib,
            "device": args.device, "batch_size": args.batch_size,
            "corpus_embeddings_cache_hit": corpus_cache_hit,
            "corpus_embeddings_cache_path": str(cache_path.relative_to(ROOT)),
            "query_embeddings_cache_hit": query_cache_hit,
            "query_embeddings_cache_path": str(baseline_path.relative_to(ROOT)) if query_cache_hit else None,
            "candidate_prediction_json": "artifacts/submission/apps_retrieval_test_top10_variant_b.json",
            "submission_format": "MTEB-compatible query-id -> document-id -> cosine-score map; top 10 per query",
            "exploratory_test_set_measurement": True,
            "independently_validated": False,
        },
        "reproduction_command": "python scripts/validate_variant_b.py --device cpu --batch-size 16",
    }
    summary_path = ROOT / "artifacts/submission/final_candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Structural verification of the emitted submission payload.
    if len(predictions) != 3765 or any(len(row) != 10 for row in predictions.values()):
        raise RuntimeError("Candidate prediction artifact does not contain 3,765 top-10 lists")
    print(json.dumps({
        "NDCG@10": ndcg, "MRR@10": mrr, "runtime_seconds": runtime_seconds,
        "retrieval_and_metric_seconds": retrieval_and_metric_seconds,
        "peak_working_set_mib": peak_mib, "query_count": len(predictions),
        "top_k": 10, "submission_json": str(candidate_path),
        "summary_json": str(summary_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
