from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Iterable

import requests


WORD_RE = re.compile(r"[A-Za-z0-9']+")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    reference = words(reference_text)
    hypothesis = words(hypothesis_text)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def read_text(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def poll_metrics(base_url: str, seconds: int, interval: float) -> list[dict]:
    samples = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        response = requests.get(f"{base_url.rstrip('/')}/metrics", timeout=5)
        response.raise_for_status()
        samples.append(response.json())
        time.sleep(interval)
    return samples


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: Iterable[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = min(len(ordered) - 1, max(0, math.ceil((p / 100) * len(ordered)) - 1))
    return ordered[rank]


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting Co-Pilot runtime benchmark helper")
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--reference", help="Optional ground-truth transcript text file for WER")
    parser.add_argument("--hypothesis", default="outputs/transcript.txt", help="Transcript file to compare")
    parser.add_argument("--out", default="outputs/benchmark_report.json")
    args = parser.parse_args()

    samples = poll_metrics(args.base_url, args.seconds, args.interval)
    transcription_times = [float(item.get("last_transcription_time") or 0) for item in samples if item.get("last_transcription_time")]
    report = {
        "duration_seconds": args.seconds,
        "samples": len(samples),
        "latency": {
            "avg_transcription_seconds": round(average(transcription_times), 4),
            "p95_transcription_seconds": round(percentile(transcription_times, 95), 4),
        },
        "resources": {
            "avg_cpu_percent": round(average(float(item.get("cpu_percent") or 0) for item in samples), 2),
            "max_memory_mb": max((float(item.get("memory_mb") or 0) for item in samples), default=0),
        },
        "queues": {
            "max_audio_queue_size": max((int(item.get("audio_queue_size") or 0) for item in samples), default=0),
        },
        "throughput": {
            "captured_blocks": samples[-1].get("captured_blocks", 0) if samples else 0,
            "transcribed_chunks": samples[-1].get("transcribed_chunks", 0) if samples else 0,
            "skipped_chunks": samples[-1].get("skipped_chunks", 0) if samples else 0,
            "segments": samples[-1].get("segments", 0) if samples else 0,
        },
    }
    if args.reference:
        report["wer"] = round(
            word_error_rate(read_text(args.reference), read_text(args.hypothesis)),
            4,
        )
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
