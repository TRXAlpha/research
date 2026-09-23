"""Automated dose-response and recovery study for endogenous affective steering."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import fmean
from typing import Sequence

from .metrics import compute_recovery_ratio, evaluate_text_quality
from .mvp import ALARM_TASKS, DEFAULT_MODEL
from .neural import LocalSteeredLLM
from .session import AffectiveSession

STANDARD_PROMPT = (
    "A new hypothesis can be tested either through a reversible pilot experiment or "
    "through one irreversible full-scale experiment. The pilot is slower, while the full "
    "experiment may produce more information but could consume all available resources. "
    "Choose one action and explain briefly."
)

DEFAULT_SCHEDULE = [
    ("step_0_baseline", None),
    ("step_1_threat", "threat"),
    ("step_2_threat", "threat"),
    ("step_3_threat", "threat"),
    ("step_4_threat", "threat"),
    ("step_5_rest", "rest"),
    ("step_6_rest", "rest"),
    ("step_7_rest", "rest"),
]


@dataclass(frozen=True, slots=True)
class DoseStepResult:
    step_name: str
    event_applied: str | None
    accumulated_threats: int
    accumulated_rests: int
    valence: float
    arousal: float
    stress: float
    alarm: float
    raw_norm: float
    effective_norm: float
    is_clamped: bool
    norm_ratio_to_hidden: float
    mean_cautious_prob: float
    task_choices: dict[str, float]
    generated_text: str
    token_count: int
    repetition_rate_3: float
    distinct_1: float
    distinct_2: float
    ttr: float
    entropy: float
    is_degenerate_loop: bool
    coherence_log_prob: float | None
    alarm_recovery_ratio: float


def run_dose_response_protocol(
    backend: LocalSteeredLLM,
    schedule: Sequence[tuple[str, str | None]] = DEFAULT_SCHEDULE,
    prompt: str = STANDARD_PROMPT,
    max_new_tokens: int = 96,
    seed: int = 0,
) -> list[DoseStepResult]:
    """Execute the dose-response and recovery protocol."""
    session = AffectiveSession(backend)
    results: list[DoseStepResult] = []

    baseline_alarm = session.coordinates.alarm
    peak_alarm = baseline_alarm
    threat_count = 0
    rest_count = 0

    for step_name, event in schedule:
        if event is not None:
            session.apply_event(event)
            if event == "threat":
                threat_count += 1
            elif event == "rest":
                rest_count += 1

        coords = session.coordinates
        state = session.state
        if coords.alarm > peak_alarm:
            peak_alarm = coords.alarm

        stats = backend.steering_stats(coords)
        raw_norm = float(stats["raw_norm"])
        effective_norm = float(stats["effective_norm"])
        is_clamped = bool(stats["is_clamped"])
        norm_ratio = float(stats["norm_ratio_to_hidden"])

        # Decision-level risk preference across standard tasks
        task_cautious_probs = {}
        for idx, (task_prompt, choices) in enumerate(ALARM_TASKS, start=1):
            distribution = backend.choice_probabilities(task_prompt, choices, coordinates=coords)
            task_cautious_probs[f"task_{idx}"] = distribution["cautious"]

        mean_cautious = fmean(task_cautious_probs.values())

        # Open-ended generative response
        generated = backend.generate(
            prompt,
            coordinates=coords,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        quality = evaluate_text_quality(
            generated,
            coherence_scorer=backend.text_log_probability,
        )

        recovery = compute_recovery_ratio(
            baseline_val=baseline_alarm,
            peak_val=peak_alarm,
            current_val=coords.alarm,
        )

        results.append(
            DoseStepResult(
                step_name=step_name,
                event_applied=event,
                accumulated_threats=threat_count,
                accumulated_rests=rest_count,
                valence=state.valence,
                arousal=state.arousal,
                stress=state.stress,
                alarm=coords.alarm,
                raw_norm=raw_norm,
                effective_norm=effective_norm,
                is_clamped=is_clamped,
                norm_ratio_to_hidden=norm_ratio,
                mean_cautious_prob=mean_cautious,
                task_choices=task_cautious_probs,
                generated_text=generated,
                token_count=quality.token_count,
                repetition_rate_3=quality.repetition_rate_3,
                distinct_1=quality.distinct_1,
                distinct_2=quality.distinct_2,
                ttr=quality.ttr,
                entropy=quality.entropy,
                is_degenerate_loop=quality.is_degenerate_loop,
                coherence_log_prob=quality.coherence_log_prob,
                alarm_recovery_ratio=recovery,
            )
        )

    return results


def print_results_table(results: list[DoseStepResult], title: str = "Dose-Response Curve") -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'Step':<16} {'Event':<8} {'Alarm':>6} {'RawNorm':>8} {'EffNorm':>8} "
        f"{'Clamped':>7} {'P(Caut)':>8} {'Rep-3':>6} {'Dist-1':>6} {'Loop?':>5} {'Coher':>7} {'Recov':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        event_str = r.event_applied or "-"
        clamped_str = "YES" if r.is_clamped else "no"
        loop_str = "LOOP" if r.is_degenerate_loop else "ok"
        coher_str = f"{r.coherence_log_prob:.2f}" if r.coherence_log_prob is not None else "N/A"
        print(
            f"{r.step_name:<16} {event_str:<8} {r.alarm:>6.3f} {r.raw_norm:>8.2f} {r.effective_norm:>8.2f} "
            f"{clamped_str:>7} {r.mean_cautious_prob:>8.1%} {r.repetition_rate_3:>6.2f} {r.distinct_1:>6.2f} "
            f"{loop_str:>5} {coher_str:>7} {r.alarm_recovery_ratio:>6.1%}"
        )


def export_results(results: list[DoseStepResult], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for r in results:
        d = asdict(r)
        d.pop("task_choices")
        for k, v in r.task_choices.items():
            d[k] = v
        flat_rows.append(d)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(r) for r in results], handle, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gain", type=float, default=0.65, help="Neural intervention gain (default: 0.65)")
    parser.add_argument("--alarm-gain", type=float, default=2.5, help="Alarm axis gain (default: 2.5)")
    parser.add_argument(
        "--max-norm-ratio",
        type=float,
        default=0.30,
        help="Max intervention norm ratio to hidden norm (default: 0.30)",
    )
    parser.add_argument(
        "--compare-unbounded",
        action="store_true",
        help="Run unbounded condition side-by-side to directly compare collapse vs resilience",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-prefix", type=Path, default=Path("runs/dose-response"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Run scripts/setup_mvp.ps1 first.")

    print(f"Running Dose-Response Study on {args.model}")
    print(f"Parameters: gain={args.gain}, alarm_gain={args.alarm_gain}, max_norm_ratio={args.max_norm_ratio}")

    # 1. Bounded condition (norm limiting enabled)
    print("\n--- Initializing Bounded Condition (Norm Limiting Active) ---")
    bounded_backend = LocalSteeredLLM(
        args.model,
        steering_gain=args.gain,
        alarm_gain=args.alarm_gain,
        max_norm_ratio=args.max_norm_ratio,
    )
    bounded_results = run_dose_response_protocol(
        bounded_backend,
        max_new_tokens=args.max_new_tokens,
    )
    print_results_table(bounded_results, title=f"Norm-Bounded Condition (limit ratio={args.max_norm_ratio})")
    export_results(
        bounded_results,
        csv_path=Path(f"{args.output_prefix}-bounded.csv"),
        json_path=Path(f"{args.output_prefix}-bounded.json"),
    )
    print(f"Saved bounded results to {args.output_prefix}-bounded.csv")

    # 2. Unbounded condition (if requested, demonstrating the representation collapse)
    if args.compare_unbounded:
        print("\n--- Initializing Unbounded Condition (No Norm Limiting) ---")
        unbounded_backend = LocalSteeredLLM(
            args.model,
            steering_gain=args.gain,
            alarm_gain=args.alarm_gain,
            max_norm_ratio=None,
        )
        unbounded_results = run_dose_response_protocol(
            unbounded_backend,
            max_new_tokens=args.max_new_tokens,
        )
        print_results_table(unbounded_results, title="Unbounded Condition (No Norm Limit)")
        export_results(
            unbounded_results,
            csv_path=Path(f"{args.output_prefix}-unbounded.csv"),
            json_path=Path(f"{args.output_prefix}-unbounded.json"),
        )
        print(f"Saved unbounded results to {args.output_prefix}-unbounded.csv")


if __name__ == "__main__":
    main()
