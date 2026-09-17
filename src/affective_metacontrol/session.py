"""Persistent endogenous affect session coupled to a frozen local LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol

from .affect import ConstitutiveAffectSystem
from .events import EVENTS, apply_event_to_homeostasis, get_event
from .models import AffectState, HomeostaticState, NeuralAffectCoordinates
from .modulation import neural_coordinates


class GenerativeBackend(Protocol):
    def generate(self, prompt: str, *, coordinates=None, max_new_tokens: int = 96, seed: int = 0, sample: bool = False) -> str: ...
    def readout(self, text: str): ...


@dataclass(frozen=True, slots=True)
class Comparison:
    prompt: str
    baseline: str
    affective: str
    coordinates: NeuralAffectCoordinates
    baseline_readout: object
    affective_readout: object


class AffectiveSession:
    """Affect is constitutive; only world events are accepted as input."""

    def __init__(self, backend: GenerativeBackend, log_path: str | Path | None = None) -> None:
        self.backend = backend
        self.affect = ConstitutiveAffectSystem()
        self.homeostasis = HomeostaticState()
        self.log_path = Path(log_path) if log_path else None

    @property
    def state(self) -> AffectState:
        return self.affect.state

    @property
    def coordinates(self) -> NeuralAffectCoordinates:
        return neural_coordinates(self.state)

    def apply_event(self, name: str) -> AffectState:
        event = get_event(name)
        self.homeostasis = apply_event_to_homeostasis(self.homeostasis, event)
        state = self.affect.step(event.appraisal, self.homeostasis, cognitive_cost=0.15)
        self._log({"type": "event", "event": event.name, "state": asdict(state), "homeostasis": asdict(self.homeostasis)})
        return state

    def compare(self, prompt: str, *, max_new_tokens: int = 96, seed: int = 0) -> Comparison:
        baseline = self.backend.generate(prompt, coordinates=None, max_new_tokens=max_new_tokens, seed=seed)
        affective = self.backend.generate(
            prompt,
            coordinates=self.coordinates,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        comparison = Comparison(
            prompt=prompt,
            baseline=baseline,
            affective=affective,
            coordinates=self.coordinates,
            baseline_readout=self.backend.readout(baseline),
            affective_readout=self.backend.readout(affective),
        )
        self._log(
            {
                "type": "comparison",
                "prompt": prompt,
                "coordinates": asdict(comparison.coordinates),
                "baseline": baseline,
                "affective": affective,
                "baseline_readout": asdict(comparison.baseline_readout),
                "affective_readout": asdict(comparison.affective_readout),
            }
        )
        return comparison

    def reset(self) -> None:
        self.affect.reset()
        self.homeostasis = HomeostaticState()
        self._log({"type": "reset"})

    def snapshot(self) -> dict[str, object]:
        return {
            "affect": asdict(self.state),
            "homeostasis": asdict(self.homeostasis),
            "neural_coordinates": asdict(self.coordinates),
        }

    def _log(self, payload: dict[str, object]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def available_events() -> dict[str, str]:
    return {name: event.description for name, event in sorted(EVENTS.items())}

