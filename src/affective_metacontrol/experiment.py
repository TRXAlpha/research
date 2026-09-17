"""Multi-seed experiment runner and dependency-free summaries."""

from __future__ import annotations

from dataclasses import asdict
import csv
from pathlib import Path
from statistics import fmean

from .agents import EpisodeResult, HypothesisAgent, run_episode


def run_study(episodes: int, budget: int, seed: int = 0) -> list[EpisodeResult]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")

    results: list[EpisodeResult] = []
    for offset in range(episodes):
        world_seed = seed + offset
        for mode in sorted(HypothesisAgent.MODES):
            results.append(run_episode(mode=mode, seed=world_seed, budget=budget))
    return results


def summarize(results: list[EpisodeResult]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    modes = sorted({result.mode for result in results})
    for mode in modes:
        group = [result for result in results if result.mode == mode]
        summaries[mode] = {
            "episodes": float(len(group)),
            "exact_rate": fmean(float(item.exact) for item in group),
            "mean_accuracy": fmean(item.accuracy for item in group),
            "mean_experiments": fmean(item.experiments for item in group),
            "mean_remaining_hypotheses": fmean(item.remaining_hypotheses for item in group),
        }
    return summaries


def write_csv(results: list[EpisodeResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

