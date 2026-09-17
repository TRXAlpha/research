"""Constitutive affect dynamics.

The transition runs automatically after appraisal. It is deliberately not an
action available to an agent policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import AffectState, Appraisal, HomeostaticState, clamp


@dataclass(slots=True)
class AffectDynamicsConfig:
    fast_rate: float = 0.42
    mood_rate: float = 0.035
    fatigue_rate: float = 0.08
    recovery_rate: float = 0.045
    baseline_arousal: float = 0.2


class ConstitutiveAffectSystem:
    """Persistent affect updated by appraisal and synthetic homeostasis."""

    def __init__(
        self,
        state: AffectState | None = None,
        config: AffectDynamicsConfig | None = None,
    ) -> None:
        self.state = (state or AffectState()).bounded()
        self.config = config or AffectDynamicsConfig()

    @staticmethod
    def _approach(current: float, target: float, rate: float) -> float:
        return current + rate * (target - current)

    def step(
        self,
        appraisal: Appraisal,
        homeostasis: HomeostaticState,
        *,
        cognitive_cost: float = 0.0,
    ) -> AffectState:
        """Apply one unavoidable state transition and return the new state."""

        a = appraisal.bounded()
        h = homeostasis.bounded()
        previous = self.state
        cfg = self.config

        unmet = h.mean_deviation()
        surprise = clamp(0.55 * abs(a.prediction_error) + 0.45 * a.novelty, 0.0, 1.0)
        threat = clamp(
            a.urgency * (1.0 - a.controllability) * max(0.0, -a.goal_congruence),
            0.0,
            1.0,
        )

        target_valence = clamp(
            0.62 * a.goal_congruence
            + 0.23 * a.prediction_error
            + 0.15 * previous.mood_valence
            - 0.25 * unmet,
            -1.0,
            1.0,
        )
        target_arousal = clamp(
            cfg.baseline_arousal
            + 0.42 * surprise
            + 0.28 * a.urgency
            + 0.22 * unmet,
            0.0,
            1.0,
        )
        target_dominance = clamp(
            1.25 * (a.controllability - 0.5)
            + 0.25 * a.agency
            + 0.15 * (h.competence - 0.5),
            -1.0,
            1.0,
        )
        target_stress = clamp(threat + 0.48 * unmet + 0.20 * previous.fatigue, 0.0, 1.0)
        target_frustration = clamp(
            max(0.0, -a.goal_congruence) * (0.25 + 0.75 * a.controllability)
            + 0.25 * max(0.0, -a.prediction_error)
            + 0.10 * previous.frustration,
            0.0,
            1.0,
        )

        fatigue_target = clamp(
            (1.0 - h.resources) + cfg.fatigue_rate * clamp(cognitive_cost, 0.0, 1.0),
            0.0,
            1.0,
        )
        fatigue_rate = cfg.fast_rate if fatigue_target > previous.fatigue else cfg.recovery_rate

        valence = self._approach(previous.valence, target_valence, cfg.fast_rate)
        arousal = self._approach(previous.arousal, target_arousal, cfg.fast_rate)
        dominance = self._approach(previous.dominance, target_dominance, cfg.fast_rate)
        stress = self._approach(previous.stress, target_stress, cfg.fast_rate)
        frustration = self._approach(previous.frustration, target_frustration, cfg.fast_rate)
        fatigue = self._approach(previous.fatigue, fatigue_target, fatigue_rate)

        mood_valence = self._approach(previous.mood_valence, valence, cfg.mood_rate)
        mood_arousal = self._approach(previous.mood_arousal, arousal, cfg.mood_rate)

        self.state = AffectState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            stress=stress,
            frustration=frustration,
            fatigue=fatigue,
            mood_valence=mood_valence,
            mood_arousal=mood_arousal,
        ).bounded()
        return self.state

    def reset(self, state: AffectState | None = None) -> AffectState:
        self.state = (state or AffectState()).bounded()
        return self.state

