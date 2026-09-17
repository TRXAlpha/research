"""Map affect and appraisal to falsifiable cognitive-control predictions."""

from __future__ import annotations

import math

from .models import AffectState, Appraisal, CognitiveModulation, NeuralAffectCoordinates, clamp


def derive_modulation(state: AffectState, appraisal: Appraisal) -> CognitiveModulation:
    """Derive bounded cognitive gains, including an inverted-U arousal term."""

    s = state.bounded()
    a = appraisal.bounded()

    optimal_arousal = 0.46
    arousal_width = 0.24
    inverted_u = math.exp(-0.5 * ((s.arousal - optimal_arousal) / arousal_width) ** 2)
    affective_load = clamp(0.55 * s.stress + 0.30 * s.frustration + 0.35 * s.fatigue, 0.0, 1.0)

    exploration = clamp(
        0.18
        + 0.38 * a.novelty
        + 0.28 * (1.0 - a.certainty)
        + 0.18 * s.arousal
        + 0.12 * max(0.0, s.valence)
        - 0.35 * s.stress
        - 0.20 * s.fatigue,
        0.0,
        1.0,
    )
    verification = clamp(
        0.22 + 0.42 * (1.0 - a.certainty) + 0.32 * s.stress + 0.18 * abs(a.prediction_error),
        0.0,
        1.0,
    )
    efficiency = clamp(inverted_u * (1.0 - 0.58 * affective_load), 0.0, 1.0)
    working_memory = clamp(efficiency * (1.0 - 0.35 * s.stress), 0.0, 1.0)
    creative_divergence = clamp(
        0.18
        + 0.38 * exploration
        + 0.22 * max(0.0, s.valence)
        + 0.15 * a.novelty
        - 0.42 * s.stress**2
        - 0.24 * s.fatigue,
        0.0,
        1.0,
    )

    return CognitiveModulation(
        attention_gain=clamp(0.55 + 0.45 * inverted_u + 0.18 * s.arousal - 0.22 * s.stress, 0.0, 1.25),
        working_memory_gain=working_memory,
        memory_write_gain=clamp(
            0.20 + 0.36 * s.arousal + 0.24 * abs(s.valence) + 0.22 * abs(a.prediction_error),
            0.0,
            1.0,
        ),
        retrieval_gain=clamp(0.35 + 0.30 * abs(s.mood_valence) + 0.25 * verification, 0.0, 1.0),
        exploration=exploration,
        verification=verification,
        planning_depth=clamp(
            0.35 + 0.28 * max(0.0, s.dominance) + 0.34 * efficiency - 0.25 * s.fatigue,
            0.0,
            1.0,
        ),
        creative_divergence=creative_divergence,
        plasticity_gain=clamp(0.20 + 0.45 * abs(a.prediction_error) + 0.22 * s.arousal - 0.30 * s.stress, 0.0, 1.0),
        cognitive_efficiency=efficiency,
    )


def neural_coordinates(state: AffectState) -> NeuralAffectCoordinates:
    """Translate persistent affect into V/A/D-like neural control coordinates.

    Stress, frustration, and fatigue are not additional steering directions in
    the MVP. They alter the three write coordinates through preregisterable,
    transparent mappings.
    """

    s = state.bounded()
    return NeuralAffectCoordinates(
        valence=(s.valence + 0.30 * s.mood_valence - 0.40 * s.stress - 0.28 * s.frustration),
        arousal=(s.arousal - 0.20 + 0.35 * s.stress + 0.22 * s.frustration - 0.35 * s.fatigue),
        dominance=(s.dominance - 0.38 * s.stress + 0.12 * s.frustration - 0.25 * s.fatigue),
    ).bounded()
