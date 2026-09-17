"""Command-line entry point for the toy study."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_study, summarize, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_study(episodes=args.episodes, budget=args.budget, seed=args.seed)
    summaries = summarize(results)

    print("mode\texact_rate\tmean_accuracy\tmean_experiments\tremaining_hypotheses")
    for mode, values in summaries.items():
        print(
            f"{mode}\t{values['exact_rate']:.3f}\t{values['mean_accuracy']:.3f}\t"
            f"{values['mean_experiments']:.2f}\t{values['mean_remaining_hypotheses']:.2f}"
        )

    if args.output is not None:
        write_csv(results, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

