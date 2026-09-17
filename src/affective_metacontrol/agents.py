"""Toy controllers used to validate experimental plumbing before LLM coupling."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random

from .affect import ConstitutiveAffectSystem
from .benchmark import CandidateRule, CausalMicroWorld, consistent_rules, default_rules
from .models import Appraisal, CognitiveModulation, HomeostaticState, clamp
from .modulation import derive_modulation


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    mode: str
    seed: int
    target_rule: str
    selected_rule: str
    accuracy: float
    exact: bool
    experiments: int
    remaining_hypotheses: int
    final_arousal: float
    final_stress: float
    final_frustration: float


class HypothesisAgent:
    """Small causal-discovery agent with swappable control dynamics."""

    MODES = {"stateless", "generic", "affective"}

    def __init__(self, mode: str, seed: int) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(self.MODES)}")
        self.mode = mode
        self.rng = random.Random(seed)
        self.rules = list(default_rules())
        self.observations: list[tuple[tuple[int, int], bool]] = []
        self.affect = ConstitutiveAffectSystem()
        self.generic_state = 0.2
        self.homeostasis = HomeostaticState()
        self.last_appraisal = Appraisal()

    def candidates(self) -> list[CandidateRule]:
        return consistent_rules(self.observations, self.rules)

    @staticmethod
    def _information_score(point: tuple[int, int], candidates: list[CandidateRule]) -> float:
        if not candidates:
            return 0.0
        positive = sum(rule(*point) for rule in candidates) / len(candidates)
        return 1.0 - abs(positive - 0.5) * 2.0

    def _modulation(self) -> CognitiveModulation:
        if self.mode == "affective":
            return derive_modulation(self.affect.state, self.last_appraisal)
        if self.mode == "generic":
            g = clamp(self.generic_state, 0.0, 1.0)
            return CognitiveModulation(
                attention_gain=0.65 + 0.25 * g,
                working_memory_gain=0.75 - 0.20 * g,
                memory_write_gain=0.55 + 0.20 * g,
                retrieval_gain=0.55,
                exploration=0.28 + 0.48 * g,
                verification=0.30 + 0.45 * g,
                planning_depth=0.60,
                creative_divergence=0.35 + 0.30 * g,
                plasticity_gain=0.40 + 0.35 * g,
                cognitive_efficiency=0.75 - 0.20 * abs(g - 0.45),
            )
        return CognitiveModulation(
            attention_gain=0.75,
            working_memory_gain=0.75,
            memory_write_gain=0.60,
            retrieval_gain=0.55,
            exploration=0.45,
            verification=0.50,
            planning_depth=0.60,
            creative_divergence=0.50,
            plasticity_gain=0.50,
            cognitive_efficiency=0.75,
        )

    def choose_experiment(self, world: CausalMicroWorld) -> tuple[int, int]:
        tried = {point for point, _ in self.observations}
        available = [point for point in world.domain if point not in tried]
        if not available:
            return self.rng.choice(world.domain)

        candidates = self.candidates() or self.rules
        modulation = self._modulation()
        temperature = 0.05 + 0.85 * modulation.exploration

        scores = []
        for point in available:
            information = self._information_score(point, candidates)
            boundary_novelty = (abs(point[0] - point[1]) / 5.0) * modulation.creative_divergence
            scores.append(information + 0.18 * boundary_novelty)

        maximum = max(scores)
        weights = [math.exp((score - maximum) / temperature) for score in scores]
        return self.rng.choices(available, weights=weights, k=1)[0]

    def observe(self, point: tuple[int, int], outcome: bool) -> None:
        before = self.candidates() or self.rules
        predicted_probability = sum(rule(*point) for rule in before) / len(before)
        prediction_error = float(outcome) - predicted_probability
        self.observations.append((point, outcome))
        after = self.candidates()

        information_gain = clamp((len(before) - len(after)) / max(1, len(before)), 0.0, 1.0)
        certainty = 1.0 - min(1.0, max(0, len(after) - 1) / max(1, len(self.rules) - 1))
        novelty = 1.0 if len(self.observations) == 1 else clamp(abs(prediction_error), 0.0, 1.0)
        goal_congruence = clamp(2.0 * information_gain - 0.25, -1.0, 1.0)

        self.last_appraisal = Appraisal(
            novelty=novelty,
            goal_congruence=goal_congruence,
            controllability=0.85,
            certainty=certainty,
            agency=0.8,
            urgency=clamp(len(self.observations) / 8.0, 0.0, 1.0),
            prediction_error=prediction_error,
        )

        resource_level = clamp(self.homeostasis.resources - 0.07, 0.0, 1.0)
        self.homeostasis = replace(
            self.homeostasis,
            certainty=certainty,
            competence=clamp(0.55 + 0.45 * information_gain, 0.0, 1.0),
            progress=clamp(0.45 + 0.55 * information_gain, 0.0, 1.0),
            resources=resource_level,
        )

        if self.mode == "affective":
            self.affect.step(self.last_appraisal, self.homeostasis, cognitive_cost=0.35)
        elif self.mode == "generic":
            self.generic_state = 0.78 * self.generic_state + 0.22 * abs(prediction_error)

    def select_rule(self) -> CandidateRule:
        candidates = self.candidates()
        if not candidates:
            return self.rng.choice(self.rules)
        modulation = self._modulation()
        if len(candidates) == 1 or self.rng.random() < modulation.verification:
            return candidates[0]
        return self.rng.choice(candidates)


def run_episode(mode: str, seed: int, budget: int = 6) -> EpisodeResult:
    world = CausalMicroWorld.generate(seed)
    agent = HypothesisAgent(mode=mode, seed=seed + 10_000)

    for _ in range(budget):
        if len(agent.candidates()) == 1:
            break
        point = agent.choose_experiment(world)
        agent.observe(point, world.experiment(point))

    selected = agent.select_rule()
    state = agent.affect.state
    return EpisodeResult(
        mode=mode,
        seed=seed,
        target_rule=world.target.name,
        selected_rule=selected.name,
        accuracy=world.accuracy(selected),
        exact=selected.name == world.target.name,
        experiments=len(agent.observations),
        remaining_hypotheses=len(agent.candidates()),
        final_arousal=state.arousal if mode == "affective" else 0.0,
        final_stress=state.stress if mode == "affective" else 0.0,
        final_frustration=state.frustration if mode == "affective" else 0.0,
    )

