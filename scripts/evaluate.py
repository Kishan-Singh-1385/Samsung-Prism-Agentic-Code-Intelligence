"""Run the CPU-friendly Sentence Transformers baseline on MTEB AppsRetrieval."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOG = logging.getLogger("prism.evaluate")
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="Override the configured Sentence Transformers model.")
    parser.add_argument("--model-revision", help="Override the configured model revision.")
    parser.add_argument("--device", help="Override the configured device (for example: cpu).")
    parser.add_argument("--batch-size", type=int, help="Override the configured inference batch size.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np

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
        LOG.warning("PyTorch unavailable while setting deterministic seeds.")


def configure_local_caches() -> None:
    """Keep model, dataset, and MTEB caches under the writable project root."""
    artifact_root = ROOT / "artifacts"
    os.environ.setdefault("HF_HOME", str(artifact_root / "huggingface"))
    os.environ.setdefault("MTEB_CACHE", str(artifact_root / "mteb_cache"))


def peak_working_set_bytes() -> int | None:
    """Return the current process high-water working set on Windows."""
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
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_memory_info.restype = ctypes.c_int
    if get_memory_info(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return int(counters.PeakWorkingSetSize)
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        import yaml

        config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.error("Could not read config %s: %s: %s", args.config, type(exc).__name__, exc)
        return 1

    seed = int(config.get("project", {}).get("seed", 42))
    encoder_config = config.get("encoder", {})
    evaluation_config = config.get("evaluation", {})
    model_name = args.model or encoder_config.get(
        "model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )
    model_revision = args.model_revision or encoder_config.get("model_revision")
    device = args.device or encoder_config.get("device", "cpu")
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(encoder_config.get("batch_size", 16))
    )
    task_name = evaluation_config.get("task", "AppsRetrieval")
    output_folder = ROOT / evaluation_config.get("output_folder", "artifacts/evaluation")
    result_json = (
        output_folder
        / model_name.replace("/", "__")
        / str(model_revision or "")
        / f"{task_name}.json"
    )

    if batch_size < 1:
        LOG.error("Batch size must be a positive integer.")
        return 2

    set_seed(seed)
    LOG.info("Task: %s", task_name)
    LOG.info("Encoder: %s (revision: %s)", model_name, model_revision or "library-resolved")
    LOG.info("Device: %s; batch size: %d; seed: %d", device, batch_size, seed)
    LOG.info("MTEB output folder: %s", output_folder)

    try:
        run_start = time.perf_counter()
        configure_local_caches()
        import mteb

        from prism.retrieval import SentenceTransformerEncoder

        tasks = mteb.get_tasks(tasks=[task_name])
        if not tasks:
            raise RuntimeError(f"MTEB returned no task for {task_name!r}.")
        encoder = SentenceTransformerEncoder(
            model_name,
            device=device,
            batch_size=batch_size,
            revision=model_revision,
        )
        evaluation = mteb.MTEB(tasks=tasks)
        previous_evaluation_time = 0.0
        if result_json.exists():
            try:
                previous_evaluation_time = float(
                    json.loads(result_json.read_text(encoding="utf-8")).get(
                        "evaluation_time", 0.0
                    )
                    or 0.0
                )
            except (OSError, ValueError, json.JSONDecodeError):
                previous_evaluation_time = 0.0
        results = evaluation.run(
            encoder,
            output_folder=str(output_folder),
            overwrite_results=bool(evaluation_config.get("overwrite_results", True)),
            encode_kwargs={"batch_size": batch_size},
        )
        # MTEB 2.21 merges evaluation_time by addition when overwriting a task
        # result. Keep the output field as this run's duration, not cumulative.
        if result_json.exists() and results:
            task_time = max(
                0.0,
                float(getattr(results[0], "evaluation_time", 0.0) or 0.0)
                - previous_evaluation_time,
            )
            if task_time > 0:
                result_data = json.loads(result_json.read_text(encoding="utf-8"))
                result_data["evaluation_time"] = task_time
                temporary = result_json.with_suffix(result_json.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(result_data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(result_json)
    except Exception as exc:
        LOG.exception("Evaluation failed (%s): %s", type(exc).__name__, exc)
        LOG.error(
            "Retry after installing requirements and resolving any model/dataset "
            "download or authentication issue: python scripts/evaluate.py --device cpu --batch-size %d",
            batch_size,
        )
        return 1

    LOG.info("Evaluation completed; MTEB results: %s", results)
    LOG.info("MTEB result JSON files are written below %s", output_folder)
    elapsed = time.perf_counter() - run_start
    memory_bytes = peak_working_set_bytes()
    LOG.info(
        "Total runner time: %.2fs; peak working set: %s MiB",
        elapsed,
        f"{memory_bytes / (1024 * 1024):.2f}" if memory_bytes else "unavailable",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
