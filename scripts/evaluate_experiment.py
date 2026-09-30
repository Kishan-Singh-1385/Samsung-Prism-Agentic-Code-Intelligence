"""Run one controlled code-retrieval model/representation experiment."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOG = logging.getLogger("prism.evaluate_experiment")
DEFAULT_CONFIG = ROOT / "configs" / "experiment_code.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--representation", choices=("raw", "processed"), required=True)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def configure_local_caches() -> None:
    artifact_root = ROOT / "artifacts"
    os.environ.setdefault("HF_HOME", str(artifact_root / "huggingface"))
    os.environ.setdefault("MTEB_CACHE", str(artifact_root / "mteb_cache"))


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def peak_working_set_bytes() -> tuple[int | None, str]:
    """Read the process high-water mark without adding a profiling dependency."""
    if os.name == "nt":
        size_t = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", size_t),
                ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t),
                ("PeakPagefileUsage", size_t),
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
        ok = get_memory_info(process, ctypes.byref(counters), counters.cb)
        return (int(counters.PeakWorkingSetSize), "windows_peak_working_set") if ok else (None, "unavailable")

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        multiplier = 1 if sys.platform == "darwin" else 1024
        return int(usage * multiplier), "resource_max_rss"
    except (ImportError, AttributeError, OSError):
        return None, "unavailable"


def _dataset_revision(task: Any, config_revision: str | None) -> str | None:
    if config_revision:
        return config_revision
    dataset_meta = getattr(task.metadata, "dataset", None)
    if isinstance(dataset_meta, dict):
        return dataset_meta.get("revision")
    return getattr(dataset_meta, "revision", None) or getattr(task.metadata, "revision", None)


def _prepare_task(task: Any, representation: str, preprocessing: dict[str, Any]) -> dict[str, int]:
    task.load_data()
    stats = {
        "corpus_documents": 0,
        "titles_included": 0,
        "starter_code_available": 0,
        "starter_code_included": 0,
        "starter_code_omitted_as_redundant": 0,
        "queries": 0,
        "queries_changed_by_normalization": 0,
    }
    splits = set(task.metadata.eval_splits)
    if representation == "raw":
        for subset_data in task.dataset.values():
            if not isinstance(subset_data, dict):
                continue
            for split_name, payload in subset_data.items():
                if split_name not in splits or not isinstance(payload, dict):
                    continue
                corpus = payload["corpus"]
                queries = payload["queries"]
                stats["corpus_documents"] += len(corpus)
                stats["queries"] += len(queries)
                for row in corpus:
                    meta = row.get("meta_information") or {}
                    starter = meta.get("starter_code") if isinstance(meta, dict) else ""
                    stats["starter_code_available"] += int(bool(str(starter or "").strip()))
        return stats

    from prism.preprocessing import code_document_representation, normalize_query

    threshold = float(preprocessing.get("starter_overlap_threshold", 0.8))
    for subset_data in task.dataset.values():
        if not isinstance(subset_data, dict):
            continue
        for split_name, payload in subset_data.items():
            if split_name not in splits or not isinstance(payload, dict):
                continue
            corpus = payload["corpus"]
            queries = payload["queries"]

            # Dataset.map may execute callbacks in worker processes, so collect
            # counters directly from the source rows rather than mutating the
            # parent process's stats dictionary inside the map callback.
            from prism.preprocessing import starter_is_redundant

            for row in corpus:
                stats["corpus_documents"] += 1
                title = str(row.get("title") or "").strip()
                stats["titles_included"] += int(bool(title))
                meta = row.get("meta_information") or {}
                starter = str(meta.get("starter_code") or "").strip() if isinstance(meta, dict) else ""
                if starter:
                    stats["starter_code_available"] += 1
                    if starter_is_redundant(
                        starter,
                        str(row.get("text") or ""),
                        threshold,
                    ):
                        stats["starter_code_omitted_as_redundant"] += 1
                    else:
                        stats["starter_code_included"] += 1

            def process_document(row: dict[str, Any]) -> dict[str, str]:
                representation_text = code_document_representation(
                    row,
                    include_title=bool(preprocessing.get("include_title_when_nonempty", True)),
                    include_language=bool(preprocessing.get("include_language", True)),
                    include_distinct_starter=bool(
                        preprocessing.get("include_distinct_starter_code", True)
                    ),
                    starter_overlap_threshold=threshold,
                )
                return {"text": representation_text}

            payload["corpus"] = corpus.map(
                process_document,
                desc=f"Build code representations ({split_name}/{representation})",
            )

            def process_query(row: dict[str, Any]) -> dict[str, str]:
                original = str(row.get("text") or "")
                normalized = (
                    normalize_query(original)
                    if preprocessing.get("normalize_query_whitespace", True)
                    else original
                )
                stats["queries"] += 1
                stats["queries_changed_by_normalization"] += int(normalized != original)
                update = {"text": normalized}
                if preprocessing.get("preserve_original_query_column", True):
                    update["original_text"] = original
                return update

            payload["queries"] = queries.map(
                process_query,
                desc=f"Normalize queries ({split_name}/{representation})",
            )
    return stats


def _load_result_json(output_dir: Path, task_name: str) -> tuple[Path, dict[str, Any]]:
    candidates = list(output_dir.rglob(f"{task_name}.json"))
    if not candidates:
        raise FileNotFoundError(f"MTEB did not write {task_name}.json below {output_dir}")
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        import yaml

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.error("Could not read config %s: %s: %s", args.config, type(exc).__name__, exc)
        return 1

    experiment_config = config.get("experiment", {})
    model_config = config.get("model", {})
    runtime_config = config.get("runtime", {})
    preprocessing = config.get("preprocessing", {})
    task_name = str(experiment_config.get("task", "AppsRetrieval"))
    model_name = str(model_config["name"])
    model_revision = model_config.get("revision")
    device = args.device or str(runtime_config.get("device", "cpu"))
    batch_size = args.batch_size or int(runtime_config.get("batch_size", 16))
    seed = int(runtime_config.get("seed", 42))
    if batch_size < 1:
        LOG.error("Batch size must be a positive integer.")
        return 2

    configure_local_caches()
    set_seed(seed)
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    memory_source = "unavailable"
    try:
        import mteb
        from sentence_transformers import SentenceTransformer

        tasks = mteb.get_tasks(tasks=[task_name])
        if not tasks:
            raise RuntimeError(f"MTEB returned no task named {task_name!r}.")
        task = tasks[0]
        dataset_revision = _dataset_revision(task, experiment_config.get("dataset_revision"))
        preprocessing_stats = _prepare_task(task, args.representation, preprocessing)

        LOG.info("Experiment: %s_%s", experiment_config.get("name_prefix", "code"), args.representation)
        LOG.info("Task: %s; dataset revision: %s", task_name, dataset_revision)
        LOG.info("Model: %s; revision: %s", model_name, model_revision or "resolve default")
        LOG.info("Representation: %s; device: %s; batch size: %d", args.representation, device, batch_size)

        model = SentenceTransformer(
            model_name,
            device=device,
            revision=model_revision,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
        model.max_seq_length = int(model_config.get("max_seq_length", 512))
        output_root = ROOT / experiment_config.get("output_folder", "artifacts/experiments/mteb")
        output_dir = output_root / args.representation
        evaluation = mteb.MTEB(tasks=tasks)
        evaluation.run(
            model,
            output_folder=str(output_dir),
            encode_kwargs={"batch_size": batch_size},
        )

        result_path, result_json = _load_result_json(output_dir, task_name)
        split = str(getattr(task.metadata, "eval_splits", ["test"])[0])
        score_entries = result_json["scores"][split]
        score_entry = score_entries[0] if isinstance(score_entries, list) else score_entries["default"]
        model_meta_path = result_path.parent / "model_meta.json"
        model_meta = json.loads(model_meta_path.read_text(encoding="utf-8")) if model_meta_path.exists() else {}
        wall_seconds = time.perf_counter() - wall_start
        memory_bytes, memory_source = peak_working_set_bytes()
        record = {
            "experiment_name": f"{experiment_config.get('name_prefix', 'code')}_{args.representation}",
            "started_at_utc": started_at,
            "model": model_name,
            "model_revision": model_meta.get("revision", model_revision),
            "preprocessing_configuration": (
                preprocessing if args.representation == "processed" else {"mode": "raw MTEB fields"}
            ),
            "preprocessing_statistics": preprocessing_stats,
            "dataset_task": task_name,
            "dataset_revision": dataset_revision,
            "split": split,
            "corpus_size": int(preprocessing_stats["corpus_documents"] or 8765),
            "query_count": int(preprocessing_stats["queries"] or 3765),
            "ndcg_at_10": score_entry["ndcg_at_10"],
            "mrr_at_10": score_entry["mrr_at_10"],
            "runtime_seconds": wall_seconds,
            "mteb_evaluation_seconds": result_json.get("evaluation_time"),
            "device": device,
            "batch_size": batch_size,
            "max_seq_length": model.max_seq_length,
            "peak_memory_bytes": memory_bytes,
            "peak_memory_mib": round(memory_bytes / (1024 * 1024), 2) if memory_bytes else None,
            "peak_memory_measurement": memory_source,
            "mteb_result_json": str(result_path.relative_to(ROOT)),
        }

        results_path = ROOT / experiment_config.get("results_file", "artifacts/experiments/results.json")
        if results_path.exists():
            results_doc = json.loads(results_path.read_text(encoding="utf-8"))
        else:
            results_doc = {"experiments": []}
        results_doc["experiments"] = [
            existing
            for existing in results_doc.get("experiments", [])
            if existing.get("experiment_name") != record["experiment_name"]
        ]
        results_doc["experiments"].append(record)
        _atomic_write_json(results_path, results_doc)
        LOG.info("Experiment result: %s", json.dumps(record, ensure_ascii=False))
        LOG.info("Wrote experiment log to %s", results_path)
        return 0
    except Exception as exc:
        LOG.exception("Experiment failed (%s): %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
