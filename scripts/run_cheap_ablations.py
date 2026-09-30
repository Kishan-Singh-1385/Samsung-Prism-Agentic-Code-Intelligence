"""Run bounded, unsupervised MiniLM representation/query ablations on AppsRetrieval."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts" / "huggingface"))
os.environ.setdefault("MTEB_CACHE", str(ROOT / "artifacts" / "mteb_cache"))

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATASET_REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"
BATCH_SIZE = 16


def peak_memory_mib() -> float | None:
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory_info.restype = wintypes.BOOL
    if get_memory_info(get_process(), ctypes.byref(counters), counters.cb):
        return counters.PeakWorkingSetSize / (1024 * 1024)
    return None


def metric(rankings: list[list[tuple[str, float]]], qids: list[str], qrels: dict[str, dict[str, int]]) -> tuple[float, float]:
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores
    result = {qid: {doc: float(score) for doc, score in rows} for qid, rows in zip(qids, rankings, strict=True)}
    scores = calculate_retrieval_scores(result, qrels, [10])
    return float(scores.ndcg["NDCG@10"]), float(scores.mrr["MRR@10"])


def rankings(corpus: np.ndarray, queries: np.ndarray, docids: list[str], depth: int) -> list[list[tuple[str, float]]]:
    cn = corpus / np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-12)
    qn = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    desc = sorted(range(len(docids)), key=lambda i: docids[i], reverse=True)
    tie = np.empty(len(docids), dtype=np.int64)
    tie[desc] = np.arange(len(docids))
    output = []
    for q in qn:
        scores = cn @ q
        idx = np.lexsort((tie, -scores))[:depth]
        output.append([(docids[int(i)], float(scores[i])) for i in idx])
    return output


def corpus_raw(row: dict[str, Any]) -> str:
    return " ".join(str(row[k]) for k in ("title", "text") if row.get(k) not in (None, ""))


def main() -> int:
    import mteb
    import yaml
    from prism.preprocessing.code_representation import code_document_representation
    from prism.preprocessing.query_text import normalize_query
    from prism.retrieval.bm25 import BM25Retriever

    wall = time.perf_counter()
    task = mteb.get_tasks(tasks=["AppsRetrieval"])[0]
    task.load_data()
    payload = task.dataset[next(iter(task.dataset))]["test"]
    docs = list(payload["corpus"])
    queries = list(payload["queries"])
    docids = [str(x["id"]) for x in docs]
    qids = [str(x["id"]) for x in queries]
    raw_docs = [corpus_raw(x) for x in docs]
    raw_queries = [str(x.get("text") or "") for x in queries]
    qrels = {str(q): {str(d): int(r) for d, r in rel.items()} for q, rel in payload["relevant_docs"].items()}
    print("schema_counts", len(docs), len(queries), len(qrels))
    print("sample_metadata", json.dumps(docs[0].get("meta_information"), ensure_ascii=False)[:500])
    languages = sorted({str(x.get("language")) for x in docs if x.get("language")})
    print("observed_languages", languages)
    print("sample_query", raw_queries[0][:300])
    print("sample_corpus", json.dumps({k: docs[0].get(k) for k in ("title", "text", "language", "meta_information", "url")}, ensure_ascii=False)[:1000])

    cache = ROOT / "artifacts/experiments/cache/apps_test_minilm_9dc7b770b5cc6fb9.npz"
    with np.load(cache, allow_pickle=False) as z:
        if z["corpus_ids"].tolist() != docids or z["query_ids"].tolist() != qids:
            raise RuntimeError("Baseline cache IDs do not match loaded AppsRetrieval test data")
        base_corpus = z["corpus_embeddings"].copy()
        base_queries = z["query_embeddings"].copy()

    model = None
    cache_dir = ROOT / "artifacts/experiments/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []

    def record(name: str, config: str, corpus_vectors: np.ndarray, query_vectors: np.ndarray, elapsed: float, cached: bool, rank: list[list[tuple[str, float]]] | None = None) -> None:
        retrieval_start = time.perf_counter()
        if rank is None:
            rank = rankings(corpus_vectors, query_vectors, docids, 10)
        ndcg, mrr = metric(rank, qids, qrels)
        output.append({"experiment": name, "model": MODEL, "model_revision": REVISION,
                       "dataset_revision": DATASET_REVISION, "split": "test",
                       "corpus_size": len(docids), "query_count": len(qids),
                       "configuration": config, "ndcg_at_10": ndcg, "mrr_at_10": mrr,
                       "runtime_seconds": elapsed + time.perf_counter() - retrieval_start, "device": "cpu", "batch_size": BATCH_SIZE,
                       "peak_process_working_set_mib": peak_memory_mib(),
                       "cached_embeddings_used": cached,
                       "test_set_exploratory_only": True})
        print(name, f"NDCG@10={ndcg:.8f}", f"MRR@10={mrr:.8f}", f"runtime={elapsed:.2f}s", f"cache={cached}")

    start = time.perf_counter()
    record("A_official_representation_cache_control", "title + single space + text; official cached embeddings", base_corpus, base_queries, time.perf_counter() - start, True)

    from sentence_transformers import SentenceTransformer
    load_start = time.perf_counter()
    model = SentenceTransformer(MODEL, device="cpu", revision=REVISION)
    model_load = time.perf_counter() - load_start

    def encode_variant(key: str, texts: list[str]) -> tuple[np.ndarray, float, bool]:
        sig = hashlib.sha256((DATASET_REVISION + MODEL + REVISION + key).encode()).hexdigest()[:16]
        path = cache_dir / f"cheap_ablation_{sig}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if z["ids"].tolist() == docids:
                    return z["embeddings"].copy(), 0.0, True
        t = time.perf_counter()
        vec = np.asarray(model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=True), dtype=np.float32)
        with path.open("wb") as f:
            np.savez(f, ids=np.asarray(docids), embeddings=vec)
        path.with_suffix(".json").write_text(json.dumps({"representation": key, "model": MODEL, "revision": REVISION, "dataset_revision": DATASET_REVISION}, indent=2) + "\n", encoding="utf-8")
        return vec, time.perf_counter() - t, False

    language_docs = [f"python code:\n{x}" for x in raw_docs]
    vec_b, elapsed_b, cache_b = encode_variant("python_code_prefix_v1", language_docs)
    record("B_python_code_marker", "corpus='python code:\\n'+official title/text; queries raw", vec_b, base_queries, model_load + elapsed_b, cache_b)

    struct_docs = [code_document_representation(x, include_title=True, include_language=True, include_distinct_starter=False) for x in docs]
    vec_c, elapsed_c, cache_c = encode_variant("observed_language_code_sections_v1", struct_docs)
    record("C_observed_language_structural_marker", "[LANGUAGE] observed language + [CODE] text; starter_code disabled; title only when distinct; URLs excluded", vec_c, base_queries, elapsed_c, cache_c)

    t = time.perf_counter()
    norm_queries = [normalize_query(x) for x in raw_queries]
    query_changed = sum(a != b for a, b in zip(raw_queries, norm_queries, strict=True))
    norm_vectors = np.asarray(model.encode(norm_queries, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=True), dtype=np.float32)
    record("D_conservative_query_normalization", f"raw corpus vectors; normalize CR/LF, remove NUL, compress horizontal whitespace, trim lines, cap blank lines; changed {query_changed}/{len(qids)} queries", base_corpus, norm_vectors, time.perf_counter() - t, False)

    t = time.perf_counter()
    dense200 = rankings(base_corpus, base_queries, docids, 200)
    bm25_path = cache_dir / "apps_test_bm25_c6509ee25c841160.pkl"
    bm25 = BM25Retriever.load(bm25_path)
    bm25_200 = [bm25.search(q, 200) for q in raw_queries]
    fused = []
    for a, b in zip(dense200, bm25_200, strict=True):
        scores: dict[str, float] = {}
        for rows in (a, b):
            for rank, (doc, _) in enumerate(rows, 1):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (60 + rank)
        fused.append(sorted(scores.items(), key=lambda p: (p[1], p[0]), reverse=True)[:10])
    record("E_cached_RRF_depth200_k60_control", "existing BM25 index cache + baseline embedding cache; RRF over top-200 each, k=60", base_corpus, base_queries, time.perf_counter() - t, True, fused)
    ndcg, mrr = output[-1]["ndcg_at_10"], output[-1]["mrr_at_10"]
    print("E_cached_RRF_depth200_k60_control", f"NDCG@10={ndcg:.8f}", f"MRR@10={mrr:.8f}")

    report = {"notice": "Exploratory test-set ablations only; no tuning by query/qrels; official baseline untouched.",
              "model": MODEL, "model_revision": REVISION, "dataset_task": "AppsRetrieval",
              "dataset_revision": DATASET_REVISION, "split": "test", "device": "cpu",
              "batch_size": BATCH_SIZE, "records": output,
              "runner_wall_seconds": time.perf_counter() - wall,
              "peak_process_working_set_mib": peak_memory_mib()}
    out = ROOT / "artifacts/experiments/cheap_ablation_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
