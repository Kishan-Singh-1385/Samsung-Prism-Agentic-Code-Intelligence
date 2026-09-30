"""Export top-10 test rankings from the official MiniLM retrieval pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prism.retrieval.apps_index import AppsRetrievalIndex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/submission/apps_retrieval_test_top10.json",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    index = AppsRetrievalIndex.load(args.config, device=args.device)
    query_embeddings = index.test_query_embeddings()
    rankings = index.rank_query_embeddings(query_embeddings, top_k=10)
    predictions = {
        query_id: {doc_id: score for doc_id, score in ranking}
        for query_id, ranking in zip(index.query_ids, rankings, strict=True)
    }
    artifact = {
        "format": "mteb-retrieval-results-map-topk-v1",
        "task": "AppsRetrieval",
        "split": "test",
        "dataset_revision": index.dataset_revision,
        "model": index.model_name,
        "model_revision": index.model_revision,
        "similarity": "cosine",
        "top_k": 10,
        "corpus_size": len(index.corpus_ids),
        "query_count": len(index.query_ids),
        "predictions": predictions,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {len(predictions)} query rankings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
