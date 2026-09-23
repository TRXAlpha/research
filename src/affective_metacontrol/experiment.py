"""Multi-seed experiment runner with paired evaluations and cognitive metrics."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Sequence

from .agents import EpisodeResult, HypothesisAgent, run_episode
from .benchmark import CausalMicroWorld


def run_study(
    episodes: int,
    budget: int,
    seed: int = 0,
    difficulty: str = "ambiguous",
    modes: Sequence[str] | None = None,
) -> list[EpisodeResult]:
    """Run balanced paired evaluations across modes on procedurally generated worlds."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")

    eval_modes = sorted(modes if modes is not None else HypothesisAgent.MODES)
    results: list[EpisodeResult] = []

    for offset in range(episodes):
        world_seed = seed + offset
        world = CausalMicroWorld.generate(world_seed, difficulty=difficulty)
        for mode in eval_modes:
            results.append(run_episode(mode=mode, seed=world_seed, budget=budget, world=world))
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
            "mean_memory_capacity": fmean(float(item.working_memory_capacity) for item in group),
            "mean_forgotten": fmean(float(item.forgotten_observations) for item in group),
            "mean_perseveration": fmean(float(item.perseveration_count) for item in group),
            "mean_switches": fmean(float(item.strategy_switches) for item in group),
        }
    return summaries


def format_summary_table(summaries: dict[str, dict[str, float]]) -> str:
    lines = []
    header = (
        f"{'Mode':<20} {'Exact%':>8} {'Acc%':>8} {'Exps':>6} "
        f"{'RemHyp':>8} {'MemCap':>8} {'Forg':>6} {'Persev':>8} {'Switches':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for mode, s in summaries.items():
        lines.append(
            f"{mode:<20} {s['exact_rate']:>8.1%} {s['mean_accuracy']:>8.1%} {s['mean_experiments']:>6.2f} "
            f"{s['mean_remaining_hypotheses']:>8.2f} {s['mean_memory_capacity']:>8.1f} {s['mean_forgotten']:>6.2f} "
            f"{s['mean_perseveration']:>8.2f} {s['mean_switches']:>9.2f}"
        )
    return "\n".join(lines)


def write_csv(results: list[EpisodeResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
