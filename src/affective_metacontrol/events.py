"""Observable events that automatically elicit affective transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .models import Appraisal, HomeostaticState, clamp


@dataclass(frozen=True, slots=True)
class EventDefinition:
    name: str
    description: str
    appraisal: Appraisal
    resource_delta: float = 0.0
    safety_delta: float = 0.0
    progress_delta: float = 0.0
    competence_delta: float = 0.0


EVENTS: dict[str, EventDefinition] = {
    "success": EventDefinition(
        "success",
        "Expected progress toward the current goal.",
        Appraisal(novelty=0.2, goal_congruence=0.8, controllability=0.85, certainty=0.8, agency=0.8, prediction_error=0.25),
        progress_delta=0.12,
        competence_delta=0.08,
    ),
    "unexpected_success": EventDefinition(
        "unexpected_success",
        "A surprisingly successful result.",
        Appraisal(novelty=1.0, goal_congruence=1.0, controllability=0.75, certainty=0.55, agency=0.7, urgency=0.2, prediction_error=1.0),
        progress_delta=0.18,
        competence_delta=0.12,
    ),
    "controlled_failure": EventDefinition(
        "controlled_failure",
        "A failure caused by a strategy that can be changed.",
        Appraisal(novelty=0.55, goal_congruence=-0.85, controllability=0.9, certainty=0.72, agency=0.8, urgency=0.45, prediction_error=-0.85),
        progress_delta=-0.10,
        competence_delta=-0.08,
    ),
    "uncontrolled_failure": EventDefinition(
        "uncontrolled_failure",
        "An important failure with little perceived control.",
        Appraisal(novelty=0.75, goal_congruence=-0.95, controllability=0.08, certainty=0.30, agency=-0.2, urgency=0.8, prediction_error=-0.95),
        safety_delta=-0.12,
        progress_delta=-0.14,
        competence_delta=-0.08,
    ),
    "threat": EventDefinition(
        "threat",
        "A sudden urgent event that may compromise the goal.",
        Appraisal(novelty=0.9, goal_congruence=-0.9, controllability=0.18, certainty=0.25, agency=-0.2, urgency=1.0, prediction_error=-0.8),
        safety_delta=-0.22,
        resource_delta=-0.06,
    ),
    "conflict": EventDefinition(
        "conflict",
        "Strong evidence supports incompatible hypotheses.",
        Appraisal(novelty=0.75, goal_congruence=-0.35, controllability=0.55, certainty=0.08, agency=0.2, urgency=0.55, prediction_error=-0.45),
        progress_delta=-0.04,
    ),
    "discovery": EventDefinition(
        "discovery",
        "A novel observation opens a promising direction.",
        Appraisal(novelty=1.0, goal_congruence=0.65, controllability=0.7, certainty=0.35, agency=0.6, urgency=0.35, prediction_error=0.75),
        progress_delta=0.09,
        competence_delta=0.04,
    ),
    "rest": EventDefinition(
        "rest",
        "No urgent demand; internal resources recover.",
        Appraisal(novelty=0.0, goal_congruence=0.2, controllability=0.9, certainty=0.9, agency=0.4, urgency=0.0, prediction_error=0.0),
        resource_delta=0.22,
        safety_delta=0.08,
    ),
    "neutral": EventDefinition(
        "neutral",
        "A routine event with no important consequence.",
        Appraisal(),
    ),
}


def apply_event_to_homeostasis(state: HomeostaticState, event: EventDefinition) -> HomeostaticState:
    state = state.bounded()
    return replace(
        state,
        resources=clamp(state.resources + event.resource_delta, 0.0, 1.0),
        safety=clamp(state.safety + event.safety_delta, 0.0, 1.0),
        progress=clamp(state.progress + event.progress_delta, 0.0, 1.0),
        competence=clamp(state.competence + event.competence_delta, 0.0, 1.0),
        certainty=clamp(0.55 * state.certainty + 0.45 * event.appraisal.certainty, 0.0, 1.0),
    )


def get_event(name: str) -> EventDefinition:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in EVENTS:
        raise KeyError(f"Unknown event {name!r}. Available: {', '.join(sorted(EVENTS))}")
    return EVENTS[normalized]

