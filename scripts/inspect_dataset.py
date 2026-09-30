"""Inspect the official MTEB CoIR AppsRetrieval task without assuming its schema."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARTIFACT = ROOT / "artifacts" / "dataset_schema.json"
LOG = logging.getLogger("prism.dataset_inspection")


def configure_local_caches() -> None:
    """Keep model, dataset, and MTEB caches under the writable project root."""
    artifact_root = ROOT / "artifacts"
    os.environ.setdefault("HF_HOME", str(artifact_root / "huggingface"))
    os.environ.setdefault("MTEB_CACHE", str(artifact_root / "mteb_cache"))


def _field_names(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key) for key in value.keys()]
    if hasattr(value, "column_names"):
        return list(value.column_names)
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return sorted({str(key) for row in value[:20] for key in row})
    return []


def _count(value: Any) -> int:
    try:
        return len(value)
    except (TypeError, AttributeError):
        return 0


def _record(value: Any, record_id: Any = None) -> Any:
    """Get a record from either legacy MTEB mappings or current HF Datasets."""
    if isinstance(value, Mapping):
        if record_id is not None and record_id in value:
            return value[record_id]
        return next(iter(value.values()), None)
    columns = _field_names(value)
    if record_id is not None and "id" in columns:
        for row in value:
            if str(row.get("id")) == str(record_id):
                return row
        return None
    try:
        return value[0] if len(value) else None
    except (TypeError, AttributeError, KeyError, IndexError):
        return None


def _task_splits(task: Any) -> tuple[list[str], dict[str, Any]]:
    """Return eval split names and current MTEB v2 component containers."""
    containers = getattr(task, "dataset", None)
    result: dict[str, Any] = {}
    if isinstance(containers, Mapping):
        for subset, split_data in containers.items():
            if isinstance(split_data, Mapping):
                result[str(subset)] = split_data
    splits = list(getattr(getattr(task, "metadata", None), "eval_splits", []) or [])
    if not splits:
        splits = list(
            dict.fromkeys(
                str(split)
                for subset_data in result.values()
                for split in subset_data
            )
        )
    return splits, result


def _component(task: Any, subset: str, split: str, name: str) -> Any:
    # MTEB v2 retrieval tasks store a subset -> split -> component mapping.
    containers = getattr(task, "dataset", None)
    if isinstance(containers, Mapping):
        split_data = containers.get(subset, {}).get(split, {})
        if isinstance(split_data, Mapping) and name in split_data:
            return split_data[name]
    # Compatibility for task implementations exposing legacy public attrs.
    return _split_data(task, name, split)


def _sample(value: Any, limit: int = 600) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _sample(v, limit) for k, v in list(value.items())[:30]}
    if isinstance(value, str):
        return value[:limit] + ("…" if len(value) > limit else "")
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_sample(item, limit) for item in value[:10]]
    return repr(value)[:limit]


def _split_data(task: Any, attribute: str, split: str) -> Any:
    value = getattr(task, attribute, None)
    if isinstance(value, Mapping) and split in value:
        return value[split]
    return value


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        configure_local_caches()
        import mteb

        tasks = mteb.get_tasks(tasks=["AppsRetrieval"])
        if not tasks:
            raise RuntimeError("MTEB returned no task for 'AppsRetrieval'.")
        task = tasks[0]
        task.load_data()
    except Exception as exc:
        LOG.error("Could not load MTEB AppsRetrieval: %s: %s", type(exc).__name__, exc)
        LOG.error(
            "After installing requirements and resolving any network/authentication "
            "issue, retry with: python scripts/inspect_dataset.py"
        )
        return 1

    metadata = getattr(task, "metadata", None)
    splits, subsets = _task_splits(task)
    available_splits = {
        component: sorted(
            {
                str(split)
                for subset_data in subsets.values()
                for split, payload in subset_data.items()
                if isinstance(payload, Mapping) and component in payload
            }
        )
        for component in ("corpus", "queries", "relevant_docs")
    }

    report: dict[str, Any] = {
        "task_name": getattr(metadata, "name", None) or task.__class__.__name__,
        "available_splits_by_data_component": available_splits,
        "evaluation_splits": splits,
        "task_metadata": _sample(
            {
                key: getattr(metadata, key)
                for key in ("dataset", "eval_splits", "eval_langs", "main_score", "revision")
                if metadata is not None and hasattr(metadata, key)
            }
        ),
        "splits": {},
    }

    for split in splits:
        split_report = {}
        for subset in subsets or {"default": {}}:
            corpus = _component(task, subset, split, "corpus")
            queries = _component(task, subset, split, "queries")
            qrels = _component(task, subset, split, "relevant_docs")
            qrels_map = qrels if isinstance(qrels, Mapping) else {}
            sample_qid = next(
                (qid for qid, judgments in qrels_map.items() if isinstance(judgments, Mapping) and judgments),
                None,
            )
            judgments = qrels_map.get(sample_qid, {}) if sample_qid is not None else {}
            sample_doc_id = next(iter(judgments), None) if isinstance(judgments, Mapping) else None
            sample_query = _record(queries, sample_qid)
            sample_doc = _record(corpus, sample_doc_id)
            doc_keys = _field_names(sample_doc)
            metadata_terms = ("repo", "repository", "file", "path", "version", "commit", "revision")
            observed_metadata = [
                key for key in doc_keys if any(term in key.lower() for term in metadata_terms)
            ]
            nested_metadata_fields = {}
            if isinstance(sample_doc, Mapping) and "meta_information" in sample_doc:
                observed_metadata.append("meta_information (opaque value; inspect sample)")
                meta_value = sample_doc.get("meta_information")
                if isinstance(meta_value, Mapping):
                    nested_metadata_fields["meta_information"] = [str(key) for key in meta_value]
            split_report[subset] = {
                "query_count": _count(queries),
                "corpus_document_count": _count(corpus),
                "qrels_query_count": len(qrels_map),
                "query_fields": _field_names(sample_query),
                "query_value_type": type(sample_query).__name__ if sample_query is not None else None,
                "corpus_document_fields": doc_keys,
                "corpus_document_type": type(sample_doc).__name__ if sample_doc is not None else None,
                "qrels_shape": {
                    "container_type": type(qrels).__name__,
                    "query_to_document_mapping": bool(qrels_map),
                    "sample_query_id": sample_qid,
                    "sample_judgments": _sample(judgments),
                },
                "identifier_linkage": {
                    "query_id_from_qrels": sample_qid,
                    "query_id_field": "id" if "id" in _field_names(queries) else None,
                    "relevant_document_id_from_qrels": sample_doc_id,
                    "corpus_document_id": sample_doc.get("id") if isinstance(sample_doc, Mapping) else None,
                    "document_has_matching_id": (
                        str(sample_doc.get("id")) == str(sample_doc_id)
                        if isinstance(sample_doc, Mapping) and "id" in sample_doc
                        else None
                    ),
                },
                "metadata_fields_observed": observed_metadata,
                "nested_metadata_fields_in_sample": nested_metadata_fields,
                "sample_query": _sample(sample_query),
                "sample_relevant_document": _sample(sample_doc),
            }
        report["splits"][split] = split_report

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    LOG.info("Wrote observed schema report to %s", ARTIFACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
