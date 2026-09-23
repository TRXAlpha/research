"""Command-line entry point for procedural causal discovery studies."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import format_summary_table, run_study, summarize, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--budget", type=int, default=5, help="Query budget per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for world generation")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="ambiguous",
        choices=["ambiguous", "standard", "toy"],
        help="Benchmark difficulty level (default: ambiguous)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output destination")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"Running study: episodes={args.episodes}, budget={args.budget}, seed={args.seed}, difficulty={args.difficulty}")
    results = run_study(
        episodes=args.episodes,
        budget=args.budget,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    summaries = summarize(results)

    print("\n" + format_summary_table(summaries) + "\n")

    if args.output is not None:
        write_csv(results, args.output)
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
