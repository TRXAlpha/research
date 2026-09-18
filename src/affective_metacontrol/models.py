"""Typed state objects used by the affective metacontrol prototype."""

from __future__ import annotations

from dataclasses import dataclass, fields


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True, slots=True)
class Appraisal:
    """Automatic evaluation of an event relative to goals and internal state."""

    novelty: float = 0.0
    goal_congruence: float = 0.0
    controllability: float = 0.5
    certainty: float = 0.5
    agency: float = 0.0
    urgency: float = 0.0
    prediction_error: float = 0.0

    def bounded(self) -> "Appraisal":
        return Appraisal(
            novelty=clamp(self.novelty, 0.0, 1.0),
            goal_congruence=clamp(self.goal_congruence, -1.0, 1.0),
            controllability=clamp(self.controllability, 0.0, 1.0),
            certainty=clamp(self.certainty, 0.0, 1.0),
            agency=clamp(self.agency, -1.0, 1.0),
            urgency=clamp(self.urgency, 0.0, 1.0),
            prediction_error=clamp(self.prediction_error, -1.0, 1.0),
        )


@dataclass(frozen=True, slots=True)
class HomeostaticState:
    """Synthetic internal variables; one means fully satisfied."""

    certainty: float = 0.8
    competence: float = 0.8
    progress: float = 0.8
    resources: float = 0.8
    safety: float = 0.8
    affiliation: float = 0.8

    def bounded(self) -> "HomeostaticState":
        values = {item.name: clamp(getattr(self, item.name), 0.0, 1.0) for item in fields(self)}
        return HomeostaticState(**values)

    def mean_deviation(self, setpoint: float = 0.8) -> float:
        state = self.bounded()
        deviations = [max(0.0, setpoint - getattr(state, item.name)) for item in fields(state)]
        return sum(deviations) / len(deviations)


@dataclass(frozen=True, slots=True)
class AffectState:
    """Fast affect plus slow mood, stored outside visible language context."""

    valence: float = 0.0
    arousal: float = 0.2
    dominance: float = 0.0
    stress: float = 0.0
    frustration: float = 0.0
    fatigue: float = 0.0
    mood_valence: float = 0.0
    mood_arousal: float = 0.2

    def bounded(self) -> "AffectState":
        return AffectState(
            valence=clamp(self.valence, -1.0, 1.0),
            arousal=clamp(self.arousal, 0.0, 1.0),
            dominance=clamp(self.dominance, -1.0, 1.0),
            stress=clamp(self.stress, 0.0, 1.0),
            frustration=clamp(self.frustration, 0.0, 1.0),
            fatigue=clamp(self.fatigue, 0.0, 1.0),
            mood_valence=clamp(self.mood_valence, -1.0, 1.0),
            mood_arousal=clamp(self.mood_arousal, 0.0, 1.0),
        )


@dataclass(frozen=True, slots=True)
class CognitiveModulation:
    """Continuous gains emitted by the affective substrate."""

    attention_gain: float
    working_memory_gain: float
    memory_write_gain: float
    retrieval_gain: float
    exploration: float
    verification: float
    planning_depth: float
    creative_divergence: float
    plasticity_gain: float
    cognitive_efficiency: float

    def as_dict(self) -> dict[str, float]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class NeuralAffectCoordinates:
    """Coordinates sent to the neural write channel; not emotion labels."""

    valence: float
    arousal: float
    dominance: float
    alarm: float

    def bounded(self, limit: float = 1.5) -> "NeuralAffectCoordinates":
        return NeuralAffectCoordinates(
            valence=clamp(self.valence, -limit, limit),
            arousal=clamp(self.arousal, -limit, limit),
            dominance=clamp(self.dominance, -limit, limit),
            alarm=clamp(self.alarm, 0.0, limit),
        )
