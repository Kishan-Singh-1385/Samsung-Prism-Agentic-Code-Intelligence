"""Render a short MP4 from real results returned by the project retrieval index."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QUERIES = [3, 0, 2]  # Existing demo queries: input transformation, BFS, and dynamic programming.
WIDTH, HEIGHT = 1920, 1080
FPS = 30
DARK = (12, 20, 34)
PANEL = (22, 34, 53)
PANEL_ALT = (28, 43, 65)
WHITE = (239, 245, 250)
MUTED = (158, 177, 194)
TEAL = (50, 213, 190)
BLUE = (89, 168, 255)
GREEN = (122, 220, 155)


def fonts() -> dict[str, ImageFont.FreeTypeFont]:
    font_root = Path("C:/Windows/Fonts")
    candidates = {
        "regular": ("segoeui.ttf", 30),
        "medium": ("seguisb.ttf", 30),
        "bold": ("seguisb.ttf", 30),
        "mono": ("consola.ttf", 30),
    }
    sizes = {"small": 23, "body": 30, "query": 43, "title": 89, "heading": 56, "label": 24, "code": 27}
    output: dict[str, ImageFont.FreeTypeFont] = {}
    for name, (file_name, _) in candidates.items():
        for size_name, size in sizes.items():
            key = f"{name}_{size_name}"
            try:
                output[key] = ImageFont.truetype(str(font_root / file_name), size)
            except OSError:
                output[key] = ImageFont.load_default(size=size)
    return output


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int], **kwargs: Any) -> None:
    draw.text(xy, text, font=font, fill=fill, **kwargs)


def base_frame(label: str, page: str, font_map: dict[str, ImageFont.FreeTypeFont]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), DARK)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 13), fill=TEAL)
    draw_text(draw, (78, 42), "SAMSUNG PRISM 2026", font_map["medium_label"], TEAL)
    draw_text(draw, (78, 77), "AGENTIC CODE INTELLIGENCE", font_map["bold_body"], WHITE)
    draw_text(draw, (1710, 54), page, font_map["mono_small"], MUTED)
    draw.line((78, 132, 1842, 132), fill=(47, 66, 86), width=2)
    draw_text(draw, (78, 1016), label, font_map["regular_small"], MUTED)
    return image, draw


def fit_lines(text: str, chars: int, count: int) -> list[str]:
    lines = textwrap.wrap(" ".join(text.split()), width=chars, break_long_words=True, break_on_hyphens=False)
    if len(lines) > count:
        lines = lines[:count]
        lines[-1] = lines[-1].rstrip(" .") + "..."
    return lines


def title_frame(font_map: dict[str, ImageFont.FreeTypeFont], corpus: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), DARK)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 16), fill=TEAL)
    draw.rounded_rectangle((90, 180, 1830, 900), radius=38, fill=PANEL)
    draw.rounded_rectangle((144, 240, 204, 300), radius=15, fill=TEAL)
    draw_text(draw, (236, 244), "SAMSUNG PRISM 2026", font_map["medium_heading"], MUTED)
    draw_text(draw, (144, 364), "AGENTIC CODE", font_map["bold_title"], WHITE)
    draw_text(draw, (144, 470), "INTELLIGENCE", font_map["bold_title"], WHITE)
    draw.rectangle((148, 614, 550, 624), fill=TEAL)
    draw_text(draw, (144, 670), "Semantic retrieval over a real code corpus", font_map["regular_heading"], WHITE)
    draw_text(draw, (144, 760), f"{corpus:,} code snippets | CPU-based MiniLM retrieval", font_map["regular_body"], MUTED)
    draw_text(draw, (144, 846), "Natural-language questions in. Ranked code snippets out.", font_map["regular_body"], TEAL)
    return image


def context_frame(font_map: dict[str, ImageFont.FreeTypeFont], data: dict[str, Any]) -> Image.Image:
    image, draw = base_frame("SEMANTIC CODE RETRIEVAL | REAL QUERIES AND REAL RESULTS", "SYSTEM OVERVIEW", font_map)
    draw_text(draw, (78, 177), "From a programming question to ranked examples", font_map["bold_heading"], WHITE)
    blocks = [
        ("01", "NATURAL LANGUAGE", "A developer describes the code they need.", TEAL),
        ("02", "MINILM EMBEDDINGS", "The frozen model encodes query and snippets.", BLUE),
        ("03", "COSINE RANKING", "Similarity scores order the code results.", GREEN),
    ]
    for i, (number, heading, body, color) in enumerate(blocks):
        x = 78 + i * 592
        draw.rounded_rectangle((x, 300, x + 550, 515), radius=24, fill=PANEL)
        draw_text(draw, (x + 32, 326), number, font_map["mono_heading"], color)
        draw_text(draw, (x + 105, 342), heading, font_map["medium_body"], color)
        for row, line in enumerate(fit_lines(body, 39, 2)):
            draw_text(draw, (x + 32, 420 + row * 40), line, font_map["regular_body"], WHITE)
        if i < 2:
            draw_text(draw, (x + 552, 380), ">", font_map["bold_heading"], MUTED)
    draw.rounded_rectangle((78, 595, 1842, 870), radius=26, fill=PANEL_ALT)
    draw_text(draw, (120, 630), "MODEL", font_map["medium_label"], TEAL)
    draw_text(draw, (120, 676), str(data["model"]), font_map["mono_body"], WHITE)
    draw_text(draw, (120, 744), f"Revision  {data['model_revision']}", font_map["mono_small"], MUTED)
    draw_text(draw, (120, 805), f"Corpus  {data['corpus_size']:,} snippets     |     Queries  {data['query_count']:,}     |     Device  CPU", font_map["regular_body"], WHITE)
    return image


def result_frame(font_map: dict[str, ImageFont.FreeTypeFont], data: dict[str, Any], demo: dict[str, Any], index: int) -> Image.Image:
    intent = str(demo["intent"]).upper()
    image, draw = base_frame("", f"QUERY {index:02d} / 03", font_map)
    draw_text(draw, (78, 159), intent, font_map["medium_label"], TEAL)
    query_lines = fit_lines(str(demo["query"]), 82, 2)
    for row, line in enumerate(query_lines):
        draw_text(draw, (78, 198 + row * 49), line, font_map["medium_query"], WHITE)
    meta_y = 308 if len(query_lines) == 2 else 278
    latency_ms = float(demo["latency_seconds"]) * 1000
    draw.rounded_rectangle((78, meta_y, 1842, meta_y + 58), radius=16, fill=(19, 49, 58))
    draw_text(draw, (104, meta_y + 14), f"RETRIEVAL COMPLETE   {latency_ms:.2f} ms", font_map["medium_label"], TEAL)
    draw_text(draw, (1450, meta_y + 14), f"{data['corpus_size']:,} SNIPPETS", font_map["medium_label"], MUTED)

    card_y = meta_y + 74
    card_h = 104
    gap = 9
    for rank, result in enumerate(demo["results"][:5], start=1):
        y = card_y + (rank - 1) * (card_h + gap)
        draw.rounded_rectangle((78, y, 1842, y + card_h), radius=18, fill=PANEL)
        draw.rounded_rectangle((100, y + 15, 146, y + 61), radius=11, fill=(24, 65, 71))
        draw_text(draw, (111, y + 21), f"{rank}", font_map["mono_body"], TEAL)
        draw_text(draw, (168, y + 10), str(result["document_id"]), font_map["mono_body"], WHITE)
        draw_text(draw, (1624, y + 16), f"{float(result['similarity_score']):.4f}", font_map["mono_body"], GREEN)
        preview = str(result["code_preview"])
        preview_lines = [line.expandtabs(4).rstrip() for line in preview.splitlines() if line.strip()][:2]
        if not preview_lines:
            preview_lines = ["(Empty source preview)"]
        code_text = "   |   ".join(line[:106] + ("..." if len(line) > 106 else "") for line in preview_lines)
        draw_text(draw, (168, y + 57), code_text, font_map["mono_code"], MUTED)
    draw_text(draw, (78, 1000), f"TOP RESULT SCORE  {float(demo['results'][0]['similarity_score']):.4f}", font_map["medium_label"], GREEN)
    draw_text(draw, (1390, 1000), "COSINE SIMILARITY", font_map["medium_label"], MUTED)
    return image


def closing_frame(font_map: dict[str, ImageFont.FreeTypeFont], data: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), DARK)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 14), fill=TEAL)
    draw_text(draw, (112, 115), "DEMO COMPLETE", font_map["bold_title"], WHITE)
    draw_text(draw, (116, 260), "CPU-based semantic code retrieval", font_map["regular_heading"], TEAL)
    draw_text(draw, (116, 340), f"AppsRetrieval | {data['corpus_size']:,} code snippets | {data['query_count']:,} queries", font_map["regular_body"], MUTED)
    draw.rounded_rectangle((116, 465, 1804, 690), radius=26, fill=PANEL)
    draw_text(draw, (160, 512), "TOP-10 SUBMISSION", font_map["medium_heading"], TEAL)
    draw_text(draw, (160, 598), "Candidate and baseline prediction files are present in artifacts/submission/.", font_map["regular_body"], WHITE)
    draw_text(draw, (116, 812), "SAMSUNG PRISM 2026  |  AGENTIC CODE INTELLIGENCE", font_map["medium_body"], WHITE)
    return image


def write_concat_file(frame_paths: list[Path], durations: list[float], path: Path) -> None:
    entries = ["ffconcat version 1.0"]
    for frame, duration in zip(frame_paths, durations, strict=True):
        normalized = frame.resolve().as_posix().replace("'", "'\\''")
        entries.extend((f"file '{normalized}'", f"duration {duration:.3f}"))
    # Repeat final still so concat demuxer honors its duration at end of input.
    final = frame_paths[-1].resolve().as_posix().replace("'", "'\\''")
    entries.append(f"file '{final}'")
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def probe_video(ffmpeg: str, video_path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path), "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    details = proc.stderr
    duration = re.search(r"Duration: (\d+):(\d+):([\d.]+)", details)
    video = re.search(r"Video: h264[^\n]*?(\d{3,5})x(\d{3,5})", details, re.IGNORECASE)
    if proc.returncode != 0 or not duration or not video:
        raise RuntimeError("FFmpeg could not verify an H.264 video stream in the rendered MP4")
    seconds = int(duration.group(1)) * 3600 + int(duration.group(2)) * 60 + float(duration.group(3))
    if seconds < 60 or seconds > 120:
        raise RuntimeError(f"Rendered video duration outside the 60-120 second target: {seconds:.2f}s")
    return {"duration_seconds": seconds, "width": int(video.group(1)), "height": int(video.group(2)), "codec": "h264"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "demo/samsung_prism_demo.mp4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    parser.add_argument("--ffmpeg", help="FFmpeg executable; defaults to PATH or imageio-ffmpeg if installed.")
    parser.add_argument("--keep-workdir", type=Path, help="Keep rendered frames and extracted QA frames in this directory.")
    args = parser.parse_args()

    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            parser.error("FFmpeg is required to render the MP4. Install FFmpeg or supply --ffmpeg.")

    examples = json.loads((ROOT / "artifacts/demo_queries.json").read_text(encoding="utf-8"))
    selected = [examples[i] for i in QUERIES]
    print("Loading the existing AppsRetrieval MiniLM index...")

    from prism.retrieval.apps_index import AppsRetrievalIndex

    previous_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            index = AppsRetrievalIndex.load(args.config, device="cpu")
    finally:
        logging.disable(previous_level)

    demo_results = []
    for example in selected:
        actual_results, latency = index.retrieve(str(example["query"]), top_k=5)
        demo_results.append({
            "intent": str(example["intent"]),
            "query": str(example["query"]),
            "latency_seconds": float(latency),
            "results": actual_results,
        })
    actual_by_id = dict(zip(index.corpus_ids, index.corpus_texts, strict=True))
    for demo in demo_results:
        for result in demo["results"]:
            result["code_preview"] = actual_by_id.get(str(result["document_id"]), "")

    data = {
        "model": index.model_name,
        "model_revision": index.model_revision,
        "corpus_size": len(index.corpus_ids),
        "query_count": len(index.query_ids),
        "demos": demo_results,
    }
    font_map = fonts()
    durations = [5.0, 8.0, 18.0, 18.0, 18.0, 8.0]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_workdir:
        workdir = args.keep_workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="samsung_prism_video_")
        workdir = Path(cleanup.name)
    try:
        frame_dir = workdir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames = [title_frame(font_map, len(index.corpus_ids)), context_frame(font_map, data)]
        frames.extend(result_frame(font_map, data, demo, i) for i, demo in enumerate(demo_results, start=1))
        frames.append(closing_frame(font_map, data))
        frame_paths = []
        for i, frame in enumerate(frames, start=1):
            path = frame_dir / f"slide_{i:02d}.png"
            frame.save(path, format="PNG", optimize=True)
            frame_paths.append(path)

        concat_path = workdir / "frames.ffconcat"
        write_concat_file(frame_paths, durations, concat_path)
        command = [
            str(ffmpeg), "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-vf", f"fps={FPS},format=yuv420p", "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-movflags", "+faststart", "-an", str(args.output.resolve()),
        ]
        encoded = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if encoded.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed: {encoded.stderr[-2000:]}")

        verification = probe_video(str(ffmpeg), args.output.resolve())
        if verification["width"] != WIDTH or verification["height"] != HEIGHT:
            raise RuntimeError(f"Unexpected output resolution: {verification['width']}x{verification['height']}")
        qa_times = [5, 24, 42, 60]
        qa_frames = []
        for stamp in qa_times:
            frame_path = workdir / f"decoded_{stamp:02d}s.png"
            extracted = subprocess.run(
                [str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error", "-ss", str(stamp),
                 "-i", str(args.output.resolve()), "-frames:v", "1", str(frame_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            if extracted.returncode != 0 or not frame_path.is_file() or frame_path.stat().st_size == 0:
                raise RuntimeError(f"Could not extract a QA frame at {stamp}s")
            qa_frames.append(str(frame_path))
        print(json.dumps({
            "video": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            **verification,
            "query_count": len(demo_results),
            "queries": [demo["query"] for demo in demo_results],
            "retrieval_latency_ms": [round(demo["latency_seconds"] * 1000, 2) for demo in demo_results],
            "qa_frame_files": qa_frames if args.keep_workdir else [],
            "rendered_video_note": "Rendered from frames built from live retrieval results; this is not a desktop screen recording.",
        }, indent=2, ensure_ascii=False))
    finally:
        if cleanup:
            cleanup.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
