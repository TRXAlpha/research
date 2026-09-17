"""Interactive proof of concept for endogenous affective neural steering."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .neural import LocalSteeredLLM
from .session import AffectiveSession, available_events


DEFAULT_MODEL = Path("models/SmolLM2-1.7B-Instruct")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gain", type=float, default=0.85, help="Neural intervention strength")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--log", type=Path, default=Path("runs/mvp-session.jsonl"))
    parser.add_argument("--demo", action="store_true", help="Run one deterministic non-interactive demonstration")
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


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Run scripts/setup_mvp.ps1 first.")

    print("Loading frozen local model and neural calibration...")
    backend = LocalSteeredLLM(args.model, steering_gain=args.gain)
    backend.save_report("artifacts/calibration-report.json")
    session = AffectiveSession(backend, log_path=args.log)

    print("\nEndogenous Affective Metacontrol MVP")
    print("Affect is triggered by events and remains outside the visible prompt.")
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
