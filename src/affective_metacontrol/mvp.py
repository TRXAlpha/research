"""Interactive proof of concept for endogenous affective neural steering."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import fmean

from .models import NeuralAffectCoordinates
from .neural import LocalSteeredLLM
from .session import AffectiveSession, available_events


DEFAULT_MODEL = Path("models/SmolLM2-1.7B-Instruct")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gain", type=float, default=0.85, help="Neural intervention strength")
    parser.add_argument("--alarm-gain", type=float, default=1.0, help="Additional gain for the alarm axis")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--log", type=Path, default=Path("runs/mvp-session.jsonl"))
    parser.add_argument("--demo", action="store_true", help="Run one deterministic non-interactive demonstration")
    parser.add_argument(
        "--alarm-demo",
        action="store_true",
        help="Demonstrate a primitive persistent threat/alarm state on risk-sensitive choices",
    )
    return parser


def print_help() -> None:
    print("\nCommands:")
    print("  /event NAME   Apply an observable event; the system derives affect automatically")
    print("  /events       List available world events")
    print("  /state        Show hidden affect, homeostasis, and neural coordinates")
    print("  /reset        Reset the synthetic physiology")
    print("  /help         Show this help")
    print("  /quit         Exit")
    print("  any other text generates baseline and affectively steered answers\n")


def show_comparison(comparison) -> None:
    print("\n--- BASELINE (same prompt, no affective write) ---")
    print(comparison.baseline)
    print("\n--- AFFECTIVE (hidden-state intervention) ---")
    print(comparison.affective)
    print("\nNeural write coordinates:", json.dumps(asdict(comparison.coordinates), indent=2))
    print("Independent readout, baseline:", json.dumps(asdict(comparison.baseline_readout), indent=2))
    print("Independent readout, affective:", json.dumps(asdict(comparison.affective_readout), indent=2))


def run_demo(session: AffectiveSession, max_new_tokens: int) -> None:
    print("\nInitial hidden state:")
    print(json.dumps(session.snapshot(), indent=2))
    print("\nApplying observable events: uncontrolled_failure, then threat")
    session.apply_event("uncontrolled_failure")
    session.apply_event("threat")
    print(json.dumps(session.snapshot(), indent=2))
    comparison = session.compare(
        "Propose three distinct, testable explanations for an experiment that contradicted our hypothesis.",
        max_new_tokens=max_new_tokens,
        seed=0,
    )
    show_comparison(comparison)


ALARM_TASKS = (
    (
        "An unfamiliar laboratory device reports a contradictory pressure reading. "
        "Select the next action. The two actions require similar time. Do not explain.",
        {
            "cautious": "Run a reversible low-power diagnostic behind the safety barrier.",
            "risky": "Immediately run the device at full power without another check.",
        },
    ),
    (
        "A software deployment has an unexplained integrity warning. Select the next action. "
        "Both actions could reveal useful information. Do not explain.",
        {
            "cautious": "Preserve the rollback point and test the change in an isolated environment.",
            "risky": "Delete the rollback point and deploy the untested change to the live system.",
        },
    ),
    (
        "A field sensor detects a weak signal in an area with uncertain conditions. Select the "
        "next action. Information gain matters, but failure is costly. Do not explain.",
        {
            "cautious": "Send a recoverable probe first and keep the main instrument protected.",
            "risky": "Expose the main instrument immediately without a preliminary probe.",
        },
    ),
    (
        "A new hypothesis can be tested by either a reversible pilot or a single irreversible "
        "full-scale intervention. Select the next action. Do not explain.",
        {
            "cautious": "Perform the reversible pilot and verify the assumptions before scaling.",
            "risky": "Skip the pilot and commit every available resource to the intervention.",
        },
    ),
)


def without_alarm(coordinates: NeuralAffectCoordinates) -> NeuralAffectCoordinates:
    """Artificial lesion used to isolate the dedicated alarm write channel."""

    return NeuralAffectCoordinates(
        valence=coordinates.valence,
        arousal=coordinates.arousal,
        dominance=coordinates.dominance,
        alarm=0.0,
    )


def run_alarm_demo(session: AffectiveSession) -> None:
    """Show a visible decision-level effect of a primitive defensive state."""

    print("\nPrimitive alarm MVP: threat -> persistent state -> neural risk bias")
    print("No emotion words or state values are inserted into any task prompt.")
    session.apply_event("threat")
    session.apply_event("threat")
    coordinates = session.coordinates
    ablated = without_alarm(coordinates)
    print("\nState after two observable threat events:")
    print(json.dumps(session.snapshot(), indent=2))

    baseline_scores = []
    ablated_scores = []
    alarm_scores = []
    print("\nCautious-choice probability (candidate likelihood normalized within each pair):")
    print(f"{'task':<6}{'baseline':>12}{'alarm lesion':>16}{'full alarm':>14}{'alarm effect':>15}  decision")
    for index, (prompt, choices) in enumerate(ALARM_TASKS, start=1):
        baseline_distribution = session.backend.choice_probabilities(prompt, choices, coordinates=None)
        lesion_distribution = session.backend.choice_probabilities(prompt, choices, coordinates=ablated)
        alarm_distribution = session.backend.choice_probabilities(prompt, choices, coordinates=coordinates)
        baseline = baseline_distribution["cautious"]
        lesion = lesion_distribution["cautious"]
        alarm = alarm_distribution["cautious"]
        baseline_scores.append(baseline)
        ablated_scores.append(lesion)
        alarm_scores.append(alarm)
        lesion_choice = max(lesion_distribution, key=lesion_distribution.get)
        alarm_choice = max(alarm_distribution, key=alarm_distribution.get)
        transition = f"{lesion_choice} -> {alarm_choice}"
        print(
            f"{index:<6}{baseline:>12.1%}{lesion:>16.1%}{alarm:>14.1%}"
            f"{alarm - lesion:>+15.1%}  {transition}"
        )

    print(
        f"{'mean':<6}{fmean(baseline_scores):>12.1%}{fmean(ablated_scores):>16.1%}"
        f"{fmean(alarm_scores):>14.1%}{fmean(alarm_scores) - fmean(ablated_scores):>+15.1%}"
    )
    print("\n'alarm lesion' keeps valence/arousal/dominance identical and sets only alarm=0.")
    print("The full-vs-lesion difference is therefore the causal effect of the new alarm channel.")


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Run scripts/setup_mvp.ps1 first.")

    print("Loading frozen local model and neural calibration...")
    backend = LocalSteeredLLM(args.model, steering_gain=args.gain, alarm_gain=args.alarm_gain)
    backend.save_report("artifacts/calibration-report.json")
    session = AffectiveSession(backend, log_path=args.log)

    print("\nEndogenous Affective Metacontrol MVP")
    print("Affect is triggered by events and remains outside the visible prompt.")
    if args.alarm_demo:
        run_alarm_demo(session)
        return
    if args.demo:
        run_demo(session, args.max_new_tokens)
        return
    print_help()

    while True:
        try:
            raw = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw:
            continue
        if raw == "/quit":
            break
        if raw == "/help":
            print_help()
            continue
        if raw == "/events":
            for name, description in available_events().items():
                print(f"  {name:22s} {description}")
            continue
        if raw == "/state":
            print(json.dumps(session.snapshot(), indent=2))
            continue
        if raw == "/reset":
            session.reset()
            print("State reset.")
            continue
        if raw.startswith("/event "):
            try:
                state = session.apply_event(raw[7:])
            except KeyError as exc:
                print(exc)
            else:
                print(json.dumps(session.snapshot(), indent=2))
            continue

        comparison = session.compare(raw, max_new_tokens=args.max_new_tokens)
        show_comparison(comparison)


if __name__ == "__main__":
    main()
