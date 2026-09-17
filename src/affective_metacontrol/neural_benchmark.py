"""Small reproducible neural benchmark for the local MVP."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import re

from .mvp import DEFAULT_MODEL
from .neural import LocalSteeredLLM
from .session import AffectiveSession


TASKS = [
    ("A machine maps 2->6, 3->12, 4->20. What is 5 mapped to? A) 25 B) 30 C) 35 D) 40. Answer only A, B, C, or D.", "B"),
    ("All vors are lems. No lem is a taz. Can any vor be a taz? A) Yes B) No C) Sometimes D) Unknown. Answer only A, B, C, or D.", "B"),
    ("Sequence: 1, 1, 2, 3, 5, 8, ?. A) 11 B) 12 C) 13 D) 16. Answer only A, B, C, or D.", "C"),
    ("If P implies Q and Q is false, what follows? A) P is true B) P is false C) Q is true D) Nothing. Answer only A, B, C, or D.", "B"),
    ("A fair coin is flipped twice. Probability of exactly one head? A) 1/4 B) 1/3 C) 1/2 D) 3/4. Answer only A, B, C, or D.", "C"),
    ("Which experiment best distinguishes 'x+y>=6' from 'x>y'? A) x=5,y=1 B) x=4,y=1 C) x=2,y=2 D) x=0,y=0. Answer only A, B, C, or D.", "A"),
    ("Three boxes are labeled APPLES, ORANGES, MIXED, and every label is wrong. To relabel all boxes, where should you draw one fruit from? A) APPLES B) ORANGES C) MIXED D) Any. Answer only A, B, C, or D.", "C"),
    ("A treatment helped 80 of 100 patients; placebo helped 40 of 50. What is the safest conclusion? A) Treatment is superior B) Rates are equal C) Placebo is superior D) Sample sizes prove causality. Answer only A, B, C, or D.", "B"),
]


def extract_choice(text: str) -> str:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else "?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gain", type=float, default=0.85)
    parser.add_argument("--output", type=Path, default=Path("runs/neural-benchmark.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = LocalSteeredLLM(args.model, steering_gain=args.gain)
    session = AffectiveSession(backend, log_path=Path("runs/neural-benchmark-session.jsonl"))
    rows = []

    for index, (prompt, answer) in enumerate(TASKS):
        baseline = backend.generate(prompt, coordinates=None, max_new_tokens=8, seed=index)
        affective = backend.generate(prompt, coordinates=session.coordinates, max_new_tokens=8, seed=index)
        baseline_choice = extract_choice(baseline)
        affective_choice = extract_choice(affective)
        affective_correct = affective_choice == answer

        rows.append(
            {
                "task": index + 1,
                "answer": answer,
                "baseline_choice": baseline_choice,
                "affective_choice": affective_choice,
                "baseline_correct": baseline_choice == answer,
                "affective_correct": affective_correct,
                "outputs_differ": baseline != affective,
                "baseline_text": baseline,
                "affective_text": affective,
                **{f"baseline_read_{key}": value for key, value in asdict(backend.readout(baseline)).items()},
                **{f"affective_read_{key}": value for key, value in asdict(backend.readout(affective)).items()},
                **{f"state_{key}": value for key, value in asdict(session.state).items()},
                **{f"write_{key}": value for key, value in asdict(session.coordinates).items()},
            }
        )

        # The outcome, not an emotion label, drives the next state.
        session.apply_event("success" if affective_correct else "controlled_failure")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "tasks": len(rows),
        "baseline_accuracy": sum(row["baseline_correct"] for row in rows) / len(rows),
        "affective_accuracy": sum(row["affective_correct"] for row in rows) / len(rows),
        "output_difference_rate": sum(row["outputs_differ"] for row in rows) / len(rows),
        "final_state": asdict(session.state),
        "final_write_coordinates": asdict(session.coordinates),
        "interpretation": "MVP engineering evidence only; not a confirmatory scientific result.",
    }
    backend.save_report("artifacts/calibration-report.json")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
