"""One-shot evaluation of a fixed, unfitted feature scorer on AppsRetrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prism.preprocessing.code_tokens import tokenize_programming_text
from prism.retrieval.bm25 import BM25Retriever
from prism.retrieval.rerank_features import (
    FEATURE_NAMES,
    diagnostic_scores,
    extract_pair_features,
    identifier_set,
)
from evaluate_lexical import (
    atomic_write_json,
    configure_caches,
    corpus_text,
    dense_embeddings,
    evaluate_scores,
    peak_working_set_bytes,
    rank_bm25_queries,
    rank_dense_queries,
    rankings_to_scores,
    set_seed,
)

LOG = logging.getLogger("prism.evaluate_reranker_diagnostic")
DEFAULT_CONFIG = ROOT / "configs" / "experiment_reranker_diagnostic.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def rank_by_diagnostic_score(
    features: dict[str, dict[str, float | int]],
) -> tuple[list[tuple[str, float]], dict[str, dict[str, float | int]]]:
    scores = diagnostic_scores(list(features.items()))
    ranking = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return ranking, features


def diagnostic_category(
    query_id: str,
    qrels: dict[str, dict[str, int]],
    dense_ranks: dict[str, int],
    final_ranks: dict[str, int],
) -> tuple[str, str | None]:
    relevant = [doc_id for doc_id, rel in qrels.get(query_id, {}).items() if rel > 0]
    movements = [
        (doc_id, dense_ranks.get(doc_id, 10**9), final_ranks.get(doc_id, 10**9))
        for doc_id in relevant
        if doc_id in final_ranks
    ]
    for doc_id, before, after in movements:
        if after < before:
            return "improves_relevant_result", doc_id
    for doc_id, before, after in movements:
        if before < 10**9 and after > before:
            return "hurts_relevant_result", doc_id
    for doc_id, before, after in movements:
        if before == after or abs(before - after) <= 1:
            return "little_difference", doc_id
    return "little_difference", movements[0][0] if movements else None


def select_diagnostics(
    query_ids: list[str],
    query_records: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    dense_rankings: list[list[tuple[str, float]]],
    final_rankings: list[list[tuple[str, float]]],
) -> list[tuple[int, str, str | None]]:
    grouped: dict[str, list[tuple[int, str | None]]] = {
        "improves_relevant_result": [],
        "hurts_relevant_result": [],
        "little_difference": [],
    }
    for index, query_id in enumerate(query_ids):
        dense_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_rankings[index], 1)}
        final_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(final_rankings[index], 1)}
        category, relevant_id = diagnostic_category(
            query_id, qrels, dense_ranks, final_ranks
        )
        grouped[category].append((index, relevant_id))

    selected: list[tuple[int, str, str | None]] = []
    used: set[int] = set()
    for category in grouped:
        if grouped[category]:
            index, doc_id = grouped[category][0]
            selected.append((index, category, doc_id))
            used.add(index)
    # Fill to ten using query text length quantiles without relevance-based ranking.
    remaining = [index for index in range(len(query_records)) if index not in used]
    remaining.sort(key=lambda i: len(tokenize_programming_text(str(query_records[i].get("text") or ""))))
    needed = max(0, 10 - len(selected))
    if needed and remaining:
        positions = [round(i * (len(remaining) - 1) / max(1, needed - 1)) for i in range(needed)]
        for index in positions:
            if remaining[index] not in used:
                used.add(remaining[index])
                selected.append((remaining[index], "little_difference", None))
    if len(selected) < 10:
        for index in range(len(query_records)):
            if index not in used:
                selected.append((index, "little_difference", None))
                used.add(index)
                if len(selected) == 10:
                    break
    return selected[:10]


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
    experiment = config["experiment"]
    dense_config = config["dense"]
    depth = int(dense_config["candidate_depth"])
    device = str(config["runtime"]["device"])
    batch_size = int(config["runtime"]["batch_size"])
    model_name = str(dense_config["model"])
    model_revision = str(dense_config["revision"])
    total_start = time.perf_counter()

    try:
        import mteb

        data_start = time.perf_counter()
        tasks = mteb.get_tasks(tasks=[experiment["task"]])
        if not tasks:
            raise RuntimeError("MTEB did not return AppsRetrieval.")
        task = tasks[0]
        task.load_data()
        available_splits = list(task.metadata.eval_splits)
        subset = next(iter(task.dataset))
        split_map = task.dataset[subset]
        corpus_splits = [key for key, data in split_map.items() if "corpus" in data]
        query_splits = [key for key, data in split_map.items() if "queries" in data]
        qrel_splits = [key for key, data in split_map.items() if "relevant_docs" in data]
        if set(available_splits) != {"test"} or set(split_map) != {"test"}:
            raise RuntimeError(
                "The diagnostic-only workflow expects the confirmed test-only dataset; "
                f"metadata={available_splits}, loaded={list(split_map)}"
            )
        payload = split_map["test"]
        corpus_records = list(payload["corpus"])
        query_records = list(payload["queries"])
        document_ids = [str(row["id"]) for row in corpus_records]
        query_ids = [str(row["id"]) for row in query_records]
        documents = [corpus_text(row) for row in corpus_records]
        queries = [str(row.get("text") or "") for row in query_records]
        qrels = {
            str(qid): {str(doc_id): int(rel) for doc_id, rel in rels.items()}
            for qid, rels in payload["relevant_docs"].items()
        }
        dataset_load_seconds = time.perf_counter() - data_start
        LOG.info(
            "Verified splits: metadata=%s, corpus=%s, queries=%s, qrels=%s",
            available_splits, corpus_splits, query_splits, qrel_splits,
        )
        LOG.info(
            "Loaded test corpus=%d queries=%d qrels=%d; no train/dev labels are available",
            len(document_ids), len(query_ids), len(qrels),
        )

        results_path = ROOT / experiment["output_file"]
        old_results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {"experiments": []}
        previous_run = old_results.get("lexical_experiment_run", {})
        bm25_cache_name = previous_run.get("bm25_index_cache")
        bm25_cache_path = ROOT / bm25_cache_name if bm25_cache_name else None
        bm25_start = time.perf_counter()
        if bm25_cache_path and bm25_cache_path.exists():
            bm25 = BM25Retriever.load(bm25_cache_path)
            if bm25.document_ids != document_ids:
                raise RuntimeError("Cached BM25 document order does not match AppsRetrieval.")
            bm25_index_seconds = 0.0
            bm25_cache_load_seconds = time.perf_counter() - bm25_start
        else:
            bm25_config = yaml.safe_load((ROOT / "configs/experiment_bm25.yaml").read_text(encoding="utf-8"))
            options = bm25_config["bm25"]["tokenizer"]
            allowed = {key: options[key] for key in (
                "keep_whole_identifiers", "split_snake_case", "split_camel_case",
                "keep_numbers", "min_token_length",
            )}
            bm25 = BM25Retriever.build(
                document_ids, documents,
                k1=float(bm25_config["bm25"]["k1"]),
                b=float(bm25_config["bm25"]["b"]),
                epsilon=float(bm25_config["bm25"]["epsilon"]),
                tokenizer_options=allowed,
            )
            bm25_index_seconds = time.perf_counter() - bm25_start
            bm25_cache_load_seconds = 0.0
        bm25_rankings, bm25_retrieval_seconds = rank_bm25_queries(bm25, query_records, depth)

        dense_cache_key = hashlib.sha256(
            f"{experiment['dataset_revision']}|{model_name}|{model_revision}|{len(document_ids)}|{len(query_ids)}".encode()
        ).hexdigest()[:16]
        dense_cache_path = ROOT / experiment["cache_folder"] / f"apps_test_minilm_{dense_cache_key}.npz"
        (
            corpus_embeddings,
            query_embeddings,
            dense_embedding_seconds,
            dense_model_load_seconds,
            dense_cache_hit,
            dense_cache_load_seconds,
            cold_dense_embedding_seconds,
        ) = dense_embeddings(
            dense_cache_path, document_ids, query_ids, documents, queries,
            model_name, model_revision, batch_size, device,
        )
        dense_rankings, dense_search_seconds = rank_dense_queries(
            corpus_embeddings, query_embeddings, document_ids, depth
        )

        corpus_by_id = {str(row["id"]): row for row in corpus_records}
        bm25_by_query = [dict(rows) for rows in bm25_rankings]
        dense_by_query = [dict(rows) for rows in dense_rankings]
        bm25_rank_by_query = [
            {doc_id: rank for rank, (doc_id, _) in enumerate(rows, 1)}
            for rows in bm25_rankings
        ]
        dense_rank_by_query = [
            {doc_id: rank for rank, (doc_id, _) in enumerate(rows, 1)}
            for rows in dense_rankings
        ]
        derived_by_doc: dict[str, tuple[list[str], set[str], set[tuple[str, str]]]] = {}
        final_rankings: list[list[tuple[str, float]]] = []
        feature_extraction_seconds = 0.0
        scorer_seconds = 0.0
        for index, query in enumerate(queries):
            feature_start = time.perf_counter()
            candidate_ids = set(dense_by_query[index]) | set(bm25_by_query[index])
            q_tokens = tokenize_programming_text(query)
            q_identifiers = identifier_set(query)
            feature_rows: dict[str, dict[str, float | int]] = {}
            for doc_id in candidate_ids:
                if doc_id not in derived_by_doc:
                    candidate_text = str(corpus_by_id[doc_id].get("text") or "")
                    candidate_tokens = tokenize_programming_text(candidate_text)
                    derived_by_doc[doc_id] = (
                        candidate_tokens,
                        identifier_set(candidate_text),
                        set(zip(candidate_tokens, candidate_tokens[1:])),
                    )
                candidate_tokens, candidate_identifiers, candidate_bigrams = derived_by_doc[doc_id]
                feature_rows[doc_id] = extract_pair_features(
                    query,
                    str(corpus_by_id[doc_id].get("text") or ""),
                    dense_score=dense_by_query[index].get(doc_id, 0.0),
                    bm25_score=bm25_by_query[index].get(doc_id, 0.0),
                    dense_rank=dense_rank_by_query[index].get(doc_id),
                    bm25_rank=bm25_rank_by_query[index].get(doc_id),
                    query_tokens=q_tokens,
                    candidate_tokens=candidate_tokens,
                    query_identifiers=q_identifiers,
                    candidate_identifiers=candidate_identifiers,
                    candidate_bigrams=candidate_bigrams,
                )
            feature_extraction_seconds += time.perf_counter() - feature_start
            scorer_start = time.perf_counter()
            ranking, _ = rank_by_diagnostic_score(feature_rows)
            scorer_seconds += time.perf_counter() - scorer_start
            final_rankings.append(ranking)
            if (index + 1) % 250 == 0 or index + 1 == len(queries):
                LOG.info("Feature extraction: %d/%d queries", index + 1, len(queries))

        final_score_map = rankings_to_scores(query_ids, final_rankings)
        metric_start = time.perf_counter()
        ndcg, mrr, _ = evaluate_scores(final_score_map, qrels)
        metric_seconds = time.perf_counter() - metric_start

        selected = select_diagnostics(
            query_ids, query_records, qrels, dense_rankings, final_rankings
        )
        diagnostics: dict[str, Any] = {
            "dataset_task": experiment["task"],
            "dataset_revision": experiment["dataset_revision"],
            "split": "test",
            "test_labels_used_only_for_final_metrics_and_post-ranking_diagnostics": True,
            "training_or_feature_correlation_analysis": "not performed: no train/dev qrels",
            "score_formula": (
                "equal mean of per-query min-max dense cosine and BM25 scores, "
                "dense/BM25 reciprocal ranks, lexical overlap fraction, identifier "
                "overlap fraction, and exact query bigram fraction; no fitted weights"
            ),
            "queries": [],
        }
        for index, category, category_doc_id in selected:
            dense_ranks = dense_rank_by_query[index]
            bm25_ranks = bm25_rank_by_query[index]
            final_rows = final_rankings[index]
            final_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(final_rows, 1)}
            query_tokens = tokenize_programming_text(queries[index])
            query_identifiers = identifier_set(queries[index])

            def features_for(doc_id: str) -> dict[str, float | int]:
                if doc_id not in derived_by_doc:
                    text = str(corpus_by_id[doc_id].get("text") or "")
                    tokens = tokenize_programming_text(text)
                    derived_by_doc[doc_id] = (
                        tokens, identifier_set(text), set(zip(tokens, tokens[1:]))
                    )
                tokens, identifiers, bigrams = derived_by_doc[doc_id]
                return extract_pair_features(
                    queries[index], str(corpus_by_id[doc_id].get("text") or ""),
                    dense_score=dense_by_query[index].get(doc_id, 0.0),
                    bm25_score=bm25_by_query[index].get(doc_id, 0.0),
                    dense_rank=dense_ranks.get(doc_id),
                    bm25_rank=bm25_ranks.get(doc_id),
                    query_tokens=query_tokens,
                    candidate_tokens=tokens,
                    query_identifiers=query_identifiers,
                    candidate_identifiers=identifiers,
                    candidate_bigrams=bigrams,
                )

            query_entry: dict[str, Any] = {
                "query_id": query_ids[index],
                "query": queries[index],
                "behavior_vs_dense_for_relevant_candidate": category,
                "example_relevant_document_id": category_doc_id,
                "reranked_top_candidates": [],
                "example_candidate_features": features_for(category_doc_id) if category_doc_id else None,
                "example_relevant_candidate_details": None,
            }
            if category_doc_id:
                candidate_row = corpus_by_id[category_doc_id]
                query_entry["example_relevant_candidate_details"] = {
                    "document_id": category_doc_id,
                    "dense_rank": dense_ranks.get(category_doc_id),
                    "bm25_rank": bm25_ranks.get(category_doc_id),
                    "dense_score": dense_by_query[index].get(category_doc_id),
                    "bm25_score": bm25_by_query[index].get(category_doc_id, 0.0),
                    "reranker_score": dict(final_rows).get(category_doc_id),
                    "final_rank": final_ranks.get(category_doc_id),
                    "relevant": True,
                    "code_excerpt": " ".join(str(candidate_row.get("text") or "").split())[:280],
                }
            for final_rank, (doc_id, score) in enumerate(final_rows[:10], 1):
                row = corpus_by_id[doc_id]
                query_entry["reranked_top_candidates"].append({
                    "document_id": doc_id,
                    "dense_rank": dense_ranks.get(doc_id),
                    "bm25_rank": bm25_ranks.get(doc_id),
                    "dense_score": dense_by_query[index].get(doc_id),
                    "bm25_score": bm25_by_query[index].get(doc_id, 0.0),
                    "reranker_score": score,
                    "final_rank": final_rank,
                    "relevant": int(qrels.get(query_ids[index], {}).get(doc_id, 0)) > 0,
                    "code_excerpt": " ".join(str(row.get("text") or "").split())[:280],
                })
            diagnostics["queries"].append(query_entry)
        atomic_write_json(ROOT / experiment["diagnostics_file"], diagnostics)

        baseline = next(x for x in old_results["experiments"] if x["experiment_name"] == "baseline_control")
        controls = {
            name: next(x for x in old_results["experiments"] if x["experiment_name"] == name)
            for name in (
                "bm25_code_lexical",
                "hybrid_rrf_depth200_k60",
                "hybrid_rrf_depth100_k10",
            )
        }
        baseline_ndcg = float(baseline["ndcg_at_10"])
        baseline_mrr = float(baseline["mrr_at_10"])
        classification = (
            "IMPROVEMENT"
            if ndcg > baseline_ndcg and mrr > baseline_mrr
            else "WORSE"
            if ndcg < baseline_ndcg and mrr < baseline_mrr
            else "NO SIGNIFICANT IMPROVEMENT"
        )
        completed = datetime.now(timezone.utc).isoformat()
        run_name = "diagnostic_fixed_feature_scorer_v1"
        if any(x.get("experiment_name") == run_name for x in old_results["experiments"]):
            run_name = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        memory_bytes = peak_working_set_bytes()
        experiment_record = {
            "experiment_name": run_name,
            "model": "fixed diagnostic feature scorer (unfitted; no learned reranker)",
            "training_split": None,
            "training_method": "none; AppsRetrieval exposes no train/dev relevance labels",
            "feature_correlation_analysis": "not performed; only test relevance labels exist",
            "test_split_usage": "one-shot final metric evaluation and post-ranking diagnostics only",
            "candidate_depth_per_retriever": depth,
            "candidate_construction": "union of independent dense top-200 and BM25 top-200; ranks and scores retained",
            "feature_names": list(FEATURE_NAMES),
            "scoring_formula": diagnostics["score_formula"],
            "dataset_task": experiment["task"],
            "dataset_revision": experiment["dataset_revision"],
            "split": "test",
            "corpus_size": len(document_ids),
            "query_count": len(query_ids),
            "ndcg_at_10": ndcg,
            "mrr_at_10": mrr,
            "classification_vs_baseline": classification,
            "learned_reranker_status": "BLOCKED: no legitimate train/dev labels; no model was trained",
            "device": device,
            "batch_size": batch_size,
            "runtime_seconds": time.perf_counter() - total_start,
            "runtime_breakdown_seconds": {
                "dataset_loading_seconds": dataset_load_seconds,
                "bm25_index_build_seconds": bm25_index_seconds,
                "bm25_cache_load_seconds": bm25_cache_load_seconds,
                "bm25_retrieval_seconds": bm25_retrieval_seconds,
                "dense_model_load_seconds": dense_model_load_seconds,
                "dense_embedding_seconds_this_run": dense_embedding_seconds,
                "dense_embedding_cache_load_seconds": dense_cache_load_seconds,
                "cached_cold_dense_embedding_seconds_from_cache_metadata": cold_dense_embedding_seconds,
                "dense_top200_retrieval_seconds": dense_search_seconds,
                "feature_extraction_seconds": feature_extraction_seconds,
                "diagnostic_scorer_inference_seconds": scorer_seconds,
                "metric_seconds": metric_seconds,
                "dense_embedding_cache_hit": dense_cache_hit,
            },
            "peak_memory_bytes": memory_bytes,
            "peak_memory_mib": round(memory_bytes / (1024 * 1024), 2) if memory_bytes else None,
            "decision": "Do not adopt a learned reranker; obtain a labeled train/dev split before fitting or tuning one.",
            "completed_at_utc": completed,
        }
        old_results["experiments"].append(experiment_record)
        old_results["diagnostic_reranker_run"] = {
            "completed_at_utc": completed,
            "task_metadata_eval_splits": available_splits,
            "loaded_splits": list(split_map),
            "corpus_availability_by_split": corpus_splits,
            "query_availability_by_split": query_splits,
            "qrels_availability_by_split": qrel_splits,
            "counts": {"corpus": len(document_ids), "queries": len(query_ids), "qrels": len(qrels)},
            "learned_reranker_status": "BLOCKED: no train/dev labels",
            "test_metrics_for_fixed_diagnostic_scorer": {"NDCG@10": ndcg, "MRR@10": mrr},
            "metric_classification_vs_baseline": classification,
            "baseline": {"NDCG@10": baseline["ndcg_at_10"], "MRR@10": baseline["mrr_at_10"]},
            "other_controls": {
                name: {"NDCG@10": row["ndcg_at_10"], "MRR@10": row["mrr_at_10"]}
                for name, row in controls.items()
            },
            "diagnostics_file": experiment["diagnostics_file"],
        }
        atomic_write_json(results_path, old_results)
        LOG.info("Diagnostic scorer: NDCG@10 %.5f, MRR@10 %.6f", ndcg, mrr)
        LOG.info("BM25 retrieval %.2fs; dense ranking %.2fs; features %.2fs; scorer %.2fs", bm25_retrieval_seconds, dense_search_seconds, feature_extraction_seconds, scorer_seconds)
        LOG.info("Wrote append-only experiment record %s to %s", run_name, results_path)
        return 0
    except Exception as exc:
        LOG.exception("Diagnostic reranker experiment failed (%s): %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
