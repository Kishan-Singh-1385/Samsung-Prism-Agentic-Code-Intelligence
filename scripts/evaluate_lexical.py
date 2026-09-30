"""Run controlled BM25, MiniLM, and RRF retrieval experiments on AppsRetrieval."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOG = logging.getLogger("prism.evaluate_lexical")
DEFAULT_CONFIG = ROOT / "configs" / "experiment_bm25.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def configure_caches() -> None:
    artifact_root = ROOT / "artifacts"
    os.environ.setdefault("HF_HOME", str(artifact_root / "huggingface"))
    os.environ.setdefault("MTEB_CACHE", str(artifact_root / "mteb_cache"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_memory_info.restype = ctypes.c_int
    if get_memory_info(process, ctypes.byref(counters), counters.cb):
        return int(counters.PeakWorkingSetSize)
    return None


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def corpus_text(row: dict[str, Any]) -> str:
    parts = [
        str(row[key]).strip()
        for key in ("title", "text")
        if row.get(key) not in (None, "") and str(row[key]).strip()
    ]
    return " ".join(parts)


def stable_top_indices(scores: np.ndarray, id_tie_order: np.ndarray, top_k: int) -> np.ndarray:
    # The metric implementation used by MTEB resolves equal scores by document
    # ID descending. id_tie_order was assigned in that lexical order.
    return np.lexsort((id_tie_order, -scores))[: min(top_k, len(scores))]


def rankings_to_scores(
    qids: list[str], rankings: list[list[tuple[str, float]]]
) -> dict[str, dict[str, float]]:
    return {
        qid: {doc_id: float(score) for doc_id, score in rows}
        for qid, rows in zip(qids, rankings, strict=True)
    }


def evaluate_scores(
    results: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]]
) -> tuple[dict[str, float], float, float]:
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    scores = calculate_retrieval_scores(results, qrels, [10])
    return (
        scores.ndcg["NDCG@10"],
        scores.mrr["MRR@10"],
        time.perf_counter(),
    )


def rank_bm25_queries(
    retriever: Any,
    query_records: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[list[tuple[str, float]]], float]:
    start = time.perf_counter()
    rankings = []
    for index, row in enumerate(query_records, start=1):
        rankings.append(retriever.search(str(row.get("text") or ""), top_k))
        if index % 250 == 0 or index == len(query_records):
            LOG.info("BM25 query retrieval: %d/%d", index, len(query_records))
    return rankings, time.perf_counter() - start


def dense_embeddings(
    cache_path: Path,
    document_ids: list[str],
    query_ids: list[str],
    documents: list[str],
    queries: list[str],
    model_name: str,
    model_revision: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float, float, bool, float, float]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_path.with_suffix(".json")
    cache_load_start = time.perf_counter()
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(cache_path, allow_pickle=False) as values:
            corpus_ids_cached = values["corpus_ids"].tolist()
            query_ids_cached = values["query_ids"].tolist()
            if corpus_ids_cached == document_ids and query_ids_cached == query_ids:
                cache_load_seconds = time.perf_counter() - cache_load_start
                return (
                    values["corpus_embeddings"].copy(),
                    values["query_embeddings"].copy(),
                    0.0,
                    0.0,
                    True,
                    cache_load_seconds,
                    float(metadata.get("embedding_seconds", 0.0)),
                )

    from sentence_transformers import SentenceTransformer

    model_start = time.perf_counter()
    model = SentenceTransformer(
        model_name,
        device=device,
        revision=model_revision,
    )
    model_load_seconds = time.perf_counter() - model_start
    embedding_start = time.perf_counter()
    corpus_embeddings = model.encode(
        documents,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embedding_seconds = time.perf_counter() - embedding_start
    corpus_embeddings = np.asarray(corpus_embeddings, dtype=np.float32)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            corpus_ids=np.asarray(document_ids),
            query_ids=np.asarray(query_ids),
            corpus_embeddings=corpus_embeddings,
            query_embeddings=query_embeddings,
        )
    temporary.replace(cache_path)
    atomic_write_json(
        metadata_path,
        {
            "model": model_name,
            "model_revision": model_revision,
            "embedding_seconds": embedding_seconds,
            "model_load_seconds": model_load_seconds,
            "corpus_count": len(document_ids),
            "query_count": len(query_ids),
        },
    )
    del model
    return (
        corpus_embeddings,
        query_embeddings,
        embedding_seconds,
        model_load_seconds,
        False,
        0.0,
        embedding_seconds,
    )


def rank_dense_queries(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    document_ids: list[str],
    top_k: int,
    query_chunk_size: int = 64,
) -> tuple[list[list[tuple[str, float]]], float]:
    start = time.perf_counter()
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    normalized_corpus = corpus_embeddings / np.maximum(corpus_norms, 1e-12)
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    normalized_queries = query_embeddings / np.maximum(query_norms, 1e-12)
    descending_id_indices = sorted(
        range(len(document_ids)), key=lambda i: document_ids[i], reverse=True
    )
    id_tie_order = np.empty(len(document_ids), dtype=np.int64)
    id_tie_order[descending_id_indices] = np.arange(len(document_ids), dtype=np.int64)
    rankings: list[list[tuple[str, float]]] = []
    for start_index in range(0, len(query_embeddings), query_chunk_size):
        similarity_block = normalized_queries[start_index : start_index + query_chunk_size] @ normalized_corpus.T
        for scores in similarity_block:
            indices = stable_top_indices(scores, id_tie_order, top_k)
            rankings.append(
                [(document_ids[int(i)], float(scores[i])) for i in indices]
            )
    return rankings, time.perf_counter() - start


def reciprocal_rank_fusion(
    first: list[tuple[str, float]],
    second: list[tuple[str, float]],
    *,
    candidate_depth: int,
    rrf_k: int,
) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in (first[:candidate_depth], second[:candidate_depth]):
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda item: (item[1], item[0]), reverse=True)


def metric_record(
    name: str,
    retrievers_used: list[str],
    rankings: list[list[tuple[str, float]]],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    *,
    model_name: str | None,
    model_revision: str | None,
    tokenizer_config: dict[str, Any],
    bm25_config: dict[str, Any],
    dataset_revision: str,
    corpus_size: int,
    candidate_depth: int,
    rrf_k: int | None,
    timings: dict[str, float],
    batch_size: int,
    device: str,
    peak_memory: int | None,
) -> dict[str, Any]:
    score_map = rankings_to_scores(query_ids, rankings)
    metric_start = time.perf_counter()
    ndcg, mrr, _ = evaluate_scores(score_map, qrels)
    metric_seconds = time.perf_counter() - metric_start
    timing = dict(timings)
    timing["metric_seconds"] = metric_seconds
    system_runtime = sum(
        timing.get(key, 0.0)
        for key in (
            "bm25_index_build_seconds",
            "bm25_cache_load_seconds",
            "bm25_retrieval_seconds",
            "dense_model_load_seconds",
            "dense_embedding_seconds",
            "dense_cache_load_seconds",
            "dense_ranking_seconds",
            "fusion_seconds",
            "metric_seconds",
        )
    )
    return {
        "experiment_name": name,
        "retrievers_used": retrievers_used,
        "embedding_model": model_name,
        "embedding_model_revision": model_revision,
        "tokenizer": tokenizer_config,
        "bm25_parameters": bm25_config if "BM25" in retrievers_used else None,
        "candidate_depth": candidate_depth,
        "rrf_parameters": {"k": rrf_k} if rrf_k is not None else None,
        "dataset_task": "AppsRetrieval",
        "dataset_revision": dataset_revision,
        "split": "test",
        "corpus_size": corpus_size,
        "query_count": len(query_ids),
        "ndcg_at_10": ndcg,
        "mrr_at_10": mrr,
        "runtime_seconds": system_runtime,
        "runtime_breakdown_seconds": timing,
        "device": device,
        "batch_size": batch_size,
        "peak_memory_bytes": peak_memory,
        "peak_memory_mib": round(peak_memory / (1024 * 1024), 2) if peak_memory else None,
        "peak_memory_scope": "integrated experiment process high-water mark",
    }


def select_diagnostic_queries(query_records: list[dict[str, Any]]) -> list[int]:
    from prism.preprocessing.code_tokens import tokenize_programming_text

    order = sorted(
        range(len(query_records)),
        key=lambda i: len(tokenize_programming_text(str(query_records[i].get("text") or ""))),
    )
    positions = [0, 1, 2, 3, 4]
    selected = [round(position * (len(order) - 1) / 4) for position in positions]
    return [order[index] for index in selected]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        import yaml

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.error("Could not read config: %s: %s", type(exc).__name__, exc)
        return 1

    configure_caches()
    set_seed(int(config["runtime"].get("seed", 42)))
    experiment_cfg = config["experiment"]
    bm25_cfg = config["bm25"]
    tokenizer_cfg = dict(bm25_cfg["tokenizer"])
    tokenizer_options = {
        key: tokenizer_cfg[key]
        for key in (
            "keep_whole_identifiers",
            "split_snake_case",
            "split_camel_case",
            "keep_numbers",
            "min_token_length",
        )
    }
    tokenizer_record = {
        "name": "prism.preprocessing.code_tokens.tokenize_programming_text",
        **tokenizer_cfg,
    }
    bm25_parameters = {
        "implementation": bm25_cfg["implementation"],
        "k1": float(bm25_cfg["k1"]),
        "b": float(bm25_cfg["b"]),
        "epsilon": float(bm25_cfg["epsilon"]),
    }
    candidate_depth = max(
        int(bm25_cfg["candidate_depth"]),
        int(config["hybrid"]["candidate_depth"]),
        max(int(value) for value in config["hybrid"]["depth_ablation"]),
    )
    batch_size = int(config["runtime"]["batch_size"])
    device = str(config["runtime"]["device"])
    model_name = str(config["baseline_dense"]["model"])
    model_revision = str(config["baseline_dense"]["revision"])

    wall_start = time.perf_counter()
    data_start = time.perf_counter()
    try:
        import mteb
        from prism.retrieval.bm25 import BM25Retriever

        tasks = mteb.get_tasks(tasks=[experiment_cfg["task"]])
        if not tasks:
            raise RuntimeError("MTEB did not return AppsRetrieval.")
        task = tasks[0]
        task.load_data()
        subset_name = next(iter(task.dataset))
        split_name = "test"
        payload = task.dataset[subset_name][split_name]
        corpus_records = list(payload["corpus"])
        query_records = list(payload["queries"])
        document_ids = [str(row["id"]) for row in corpus_records]
        query_ids = [str(row["id"]) for row in query_records]
        documents = [corpus_text(row) for row in corpus_records]
        queries = [str(row.get("text") or "") for row in query_records]
        qrels = {
            str(qid): {str(doc_id): int(rel) for doc_id, rel in relevant.items()}
            for qid, relevant in payload["relevant_docs"].items()
        }
        dataset_load_seconds = time.perf_counter() - data_start
        dataset_revision = str(experiment_cfg["dataset_revision"])
        LOG.info(
            "Loaded %d documents, %d queries, and %d qrels for %s/%s",
            len(corpus_records), len(query_records), len(qrels), subset_name, split_name,
        )

        cache_root = ROOT / experiment_cfg["cache_folder"]
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_signature = {
            "index_version": BM25Retriever.INDEX_VERSION,
            "dataset_revision": dataset_revision,
            "tokenizer_version": tokenizer_cfg["version"],
            "tokenizer_options": tokenizer_options,
            "k1": bm25_parameters["k1"],
            "b": bm25_parameters["b"],
            "epsilon": bm25_parameters["epsilon"],
            "rank_bm25_version": importlib.metadata.version("rank-bm25"),
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_signature, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        bm25_cache_path = cache_root / f"apps_test_bm25_{cache_key}.pkl"
        bm25_cache_meta_path = bm25_cache_path.with_suffix(".json")
        if bm25_cache_path.exists():
            cache_load_start = time.perf_counter()
            retriever = BM25Retriever.load(bm25_cache_path)
            bm25_cache_load_seconds = time.perf_counter() - cache_load_start
            bm25_index_build_seconds = 0.0
            LOG.info("Loaded BM25 index cache: %s", bm25_cache_path)
        else:
            index_start = time.perf_counter()
            retriever = BM25Retriever.build(
                document_ids,
                documents,
                k1=bm25_parameters["k1"],
                b=bm25_parameters["b"],
                epsilon=bm25_parameters["epsilon"],
                tokenizer_options=tokenizer_options,
            )
            bm25_index_build_seconds = time.perf_counter() - index_start
            bm25_cache_load_seconds = 0.0
            retriever.save(bm25_cache_path)
            atomic_write_json(
                bm25_cache_meta_path,
                {**cache_signature, "index_build_seconds": bm25_index_build_seconds},
            )
            LOG.info("Built and cached BM25 index in %.2fs", bm25_index_build_seconds)

        bm25_rankings, bm25_retrieval_seconds = rank_bm25_queries(
            retriever, query_records, candidate_depth
        )
        LOG.info("BM25 retrieval for all queries took %.2fs", bm25_retrieval_seconds)

        dense_cache_key = hashlib.sha256(
            f"{dataset_revision}|{model_name}|{model_revision}|{len(document_ids)}|{len(query_ids)}".encode("utf-8")
        ).hexdigest()[:16]
        dense_cache_path = cache_root / f"apps_test_minilm_{dense_cache_key}.npz"
        (
            corpus_embeddings,
            query_embeddings,
            dense_embedding_seconds,
            dense_model_load_seconds,
            dense_cache_hit,
            dense_cache_load_seconds,
            cold_dense_embedding_seconds,
        ) = dense_embeddings(
            dense_cache_path,
            document_ids,
            query_ids,
            documents,
            queries,
            model_name,
            model_revision,
            batch_size,
            device,
        )
        dense_rankings, dense_ranking_seconds = rank_dense_queries(
            corpus_embeddings, query_embeddings, document_ids, candidate_depth
        )
        LOG.info(
            "MiniLM embedding %.2fs (cache_hit=%s), dense top-%d search %.2fs",
            dense_embedding_seconds, dense_cache_hit, candidate_depth, dense_ranking_seconds,
        )

        memory_bytes = peak_working_set_bytes()
        bm25_runtime = {
            "bm25_index_build_seconds": bm25_index_build_seconds,
            "bm25_cache_load_seconds": bm25_cache_load_seconds,
            "bm25_retrieval_seconds": bm25_retrieval_seconds,
        }
        dense_runtime = {
            "dense_model_load_seconds": dense_model_load_seconds,
            "dense_embedding_seconds": dense_embedding_seconds,
            "dense_cache_load_seconds": dense_cache_load_seconds,
            "dense_ranking_seconds": dense_ranking_seconds,
        }
        bm25_record = metric_record(
            "bm25_code_lexical",
            ["BM25"],
            bm25_rankings,
            query_ids,
            qrels,
            model_name=None,
            model_revision=None,
            tokenizer_config=tokenizer_record,
            bm25_config=bm25_parameters,
            dataset_revision=dataset_revision,
            corpus_size=len(document_ids),
            candidate_depth=candidate_depth,
            rrf_k=None,
            timings=bm25_runtime,
            batch_size=batch_size,
            device=device,
            peak_memory=memory_bytes,
        )
        dense_record = metric_record(
            "minilm_dense_reproduction",
            ["MiniLM dense"],
            dense_rankings,
            query_ids,
            qrels,
            model_name=model_name,
            model_revision=model_revision,
            tokenizer_config={"mode": "baseline raw MTEB text fields"},
            bm25_config=bm25_parameters,
            dataset_revision=dataset_revision,
            corpus_size=len(document_ids),
            candidate_depth=candidate_depth,
            rrf_k=None,
            timings=dense_runtime,
            batch_size=batch_size,
            device=device,
            peak_memory=memory_bytes,
        )

        experiment_records: list[dict[str, Any]] = [bm25_record, dense_record]
        default_depth = int(config["hybrid"]["candidate_depth"])
        default_k = int(config["hybrid"]["rrf_k"])
        ablation_specs = [
            (depth, default_k)
            for depth in config["hybrid"]["depth_ablation"]
        ]
        ablation_specs.extend(
            (default_depth, k)
            for k in config["hybrid"]["rrf_k_ablation"]
            if int(k) != default_k
        )
        hybrid_rankings_by_spec: dict[tuple[int, int], list[list[tuple[str, float]]]] = {}
        for raw_depth, raw_k in ablation_specs:
            depth, rrf_k = int(raw_depth), int(raw_k)
            fusion_start = time.perf_counter()
            fused_rankings = [
                reciprocal_rank_fusion(
                    bm25_rows,
                    dense_rows,
                    candidate_depth=depth,
                    rrf_k=rrf_k,
                )
                for bm25_rows, dense_rows in zip(bm25_rankings, dense_rankings, strict=True)
            ]
            fusion_seconds = time.perf_counter() - fusion_start
            hybrid_rankings_by_spec[(depth, rrf_k)] = fused_rankings
            metric_seconds_hint = time.perf_counter()
            fused_results = rankings_to_scores(query_ids, fused_rankings)
            fused_ndcg, fused_mrr, _ = evaluate_scores(fused_results, qrels)
            metric_seconds = time.perf_counter() - metric_seconds_hint
            experiment_records.append(
                {
                    "experiment_name": f"hybrid_rrf_depth{depth}_k{rrf_k}",
                    "retrievers_used": ["BM25", "MiniLM dense", "Reciprocal Rank Fusion"],
                    "embedding_model": model_name,
                    "embedding_model_revision": model_revision,
                    "tokenizer": tokenizer_record,
                    "bm25_parameters": bm25_parameters,
                    "candidate_depth": depth,
                    "rrf_parameters": {"k": rrf_k},
                    "dataset_task": experiment_cfg["task"],
                    "dataset_revision": dataset_revision,
                    "split": split_name,
                    "corpus_size": len(document_ids),
                    "query_count": len(query_ids),
                    "ndcg_at_10": fused_ndcg,
                    "mrr_at_10": fused_mrr,
                    "runtime_seconds": (
                        bm25_index_build_seconds
                        + bm25_cache_load_seconds
                        + bm25_retrieval_seconds
                        + dense_model_load_seconds
                        + dense_cache_load_seconds
                        + dense_embedding_seconds
                        + dense_ranking_seconds
                        + fusion_seconds
                        + metric_seconds
                    ),
                    "runtime_breakdown_seconds": {
                        **bm25_runtime,
                        **dense_runtime,
                        "fusion_seconds": fusion_seconds,
                        "metric_seconds": metric_seconds,
                        "dense_embedding_cache_hit": dense_cache_hit,
                        "cold_dense_embedding_seconds": cold_dense_embedding_seconds,
                    },
                    "device": device,
                    "batch_size": batch_size,
                    "peak_memory_bytes": memory_bytes,
                    "peak_memory_mib": round(memory_bytes / (1024 * 1024), 2) if memory_bytes else None,
                    "peak_memory_scope": "integrated experiment process high-water mark",
                }
            )

        # Keep exactly one default hybrid record clearly labeled for the decision.
        default_hybrid_name = f"hybrid_rrf_depth{default_depth}_k{default_k}"
        for record in experiment_records:
            if record["experiment_name"] == default_hybrid_name:
                record["is_default_hybrid"] = True

        diagnostics_indices = select_diagnostic_queries(query_records)
        default_hybrid_rankings = hybrid_rankings_by_spec[(default_depth, default_k)]
        diagnostics = {
            "dataset_task": experiment_cfg["task"],
            "dataset_revision": dataset_revision,
            "split": split_name,
            "diagnostic_selection": "five query statements spanning token-count quantiles",
            "top_k_per_retriever": 5,
            "queries": [],
        }
        corpus_by_id = {str(row["id"]): row for row in corpus_records}
        for query_index in diagnostics_indices:
            query = query_records[query_index]
            entry: dict[str, Any] = {
                "query_id": str(query["id"]),
                "query_text": str(query.get("text") or ""),
                "bm25": [],
                "minilm": [],
                "hybrid": [],
            }
            for key, rows in (
                ("bm25", bm25_rankings[query_index]),
                ("minilm", dense_rankings[query_index]),
                ("hybrid", default_hybrid_rankings[query_index]),
            ):
                for rank, (doc_id, score) in enumerate(rows[:5], start=1):
                    document = corpus_by_id[doc_id]
                    code = str(document.get("text") or "")
                    excerpt = " ".join(code.strip().split())[:280]
                    entry[key].append(
                        {
                            "rank": rank,
                            "document_id": doc_id,
                            "score": score,
                            "relevant": int(qrels.get(str(query["id"]), {}).get(doc_id, 0)) > 0,
                            "code_excerpt": excerpt,
                        }
                    )
            diagnostics["queries"].append(entry)

        atomic_write_json(ROOT / experiment_cfg["diagnostics_file"], diagnostics)
        results_path = ROOT / experiment_cfg["output_file"]
        if results_path.exists():
            results_doc = json.loads(results_path.read_text(encoding="utf-8"))
        else:
            results_doc = {"experiments": []}
        names = {record["experiment_name"] for record in experiment_records}
        results_doc["experiments"] = [
            record
            for record in results_doc.get("experiments", [])
            if record.get("experiment_name") not in names
        ] + experiment_records
        baseline_control = next(
            (
                record
                for record in results_doc["experiments"]
                if record.get("experiment_name") == "baseline_control"
            ),
            {"ndcg_at_10": 0.06596, "mrr_at_10": 0.05581},
        )
        hybrid_records = [
            record
            for record in experiment_records
            if record["experiment_name"].startswith("hybrid_rrf_")
        ]
        best_hybrid = max(hybrid_records, key=lambda record: record["ndcg_at_10"])
        any_both_better = any(
            record["ndcg_at_10"] > baseline_control["ndcg_at_10"]
            and record["mrr_at_10"] > baseline_control["mrr_at_10"]
            for record in hybrid_records
        )
        all_both_worse = all(
            record["ndcg_at_10"] < baseline_control["ndcg_at_10"]
            and record["mrr_at_10"] < baseline_control["mrr_at_10"]
            for record in hybrid_records
        )
        classification = (
            "IMPROVEMENT"
            if any_both_better
            else "WORSE"
            if all_both_worse
            else "NO SIGNIFICANT IMPROVEMENT"
        )
        results_doc["lexical_decision"] = {
            "classification": classification,
            "selected_candidate": best_hybrid["experiment_name"]
            if classification == "IMPROVEMENT"
            else "baseline_control",
            "best_hybrid_by_ndcg_at_10": best_hybrid["experiment_name"],
            "baseline_control_ndcg_at_10": baseline_control["ndcg_at_10"],
            "baseline_control_mrr_at_10": baseline_control["mrr_at_10"],
            "reason": (
                "At least one tested RRF configuration improved both metrics over the baseline."
                if classification == "IMPROVEMENT"
                else "Every tested RRF configuration scored below the baseline on both metrics."
                if classification == "WORSE"
                else "The tested RRF configurations had mixed metric directions relative to the baseline."
            ),
        }
        results_doc["lexical_experiment_run"] = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_task": experiment_cfg["task"],
            "dataset_revision": dataset_revision,
            "split": split_name,
            "corpus_size": len(document_ids),
            "query_count": len(query_ids),
            "qrels_query_count": len(qrels),
            "dataset_loading_seconds": dataset_load_seconds,
            "bm25_index_build_seconds": bm25_index_build_seconds,
            "bm25_cache_load_seconds": bm25_cache_load_seconds,
            "bm25_retrieval_seconds": bm25_retrieval_seconds,
            "dense_model_load_seconds": dense_model_load_seconds,
            "dense_embedding_seconds": dense_embedding_seconds,
            "dense_cache_load_seconds": dense_cache_load_seconds,
            "dense_embedding_cache_hit": dense_cache_hit,
            "cold_dense_embedding_seconds": cold_dense_embedding_seconds,
            "dense_ranking_seconds": dense_ranking_seconds,
            "integrated_runner_wall_seconds": time.perf_counter() - wall_start,
            "peak_memory_bytes": memory_bytes,
            "peak_memory_mib": round(memory_bytes / (1024 * 1024), 2) if memory_bytes else None,
            "bm25_index_cache": str(bm25_cache_path.relative_to(ROOT)),
            "dense_embedding_cache": str(dense_cache_path.relative_to(ROOT)),
            "diagnostics_file": str((ROOT / experiment_cfg["diagnostics_file"]).relative_to(ROOT)),
        }
        atomic_write_json(results_path, results_doc)
        LOG.info(
            "BM25-only: NDCG@10 %.5f MRR@10 %.6f",
            bm25_record["ndcg_at_10"], bm25_record["mrr_at_10"],
        )
        LOG.info(
            "MiniLM-only: NDCG@10 %.5f MRR@10 %.6f",
            dense_record["ndcg_at_10"], dense_record["mrr_at_10"],
        )
        for record in experiment_records[2:]:
            LOG.info(
                "%s: NDCG@10 %.5f MRR@10 %.6f runtime %.2fs",
                record["experiment_name"],
                record["ndcg_at_10"],
                record["mrr_at_10"],
                record["runtime_seconds"],
            )
        LOG.info("Wrote result log to %s", results_path)
        LOG.info("Wrote diagnostics to %s", ROOT / experiment_cfg["diagnostics_file"])
        return 0
    except Exception as exc:
        LOG.exception("Lexical retrieval experiment failed (%s): %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
