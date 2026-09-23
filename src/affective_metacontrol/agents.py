"""Controllers with cognitive-control dynamics, working memory gating, and perseveration.

Key features:
  - Salience-gated working memory with bounded capacity.
  - Perseveration tracking (low plasticity → failure to switch hypothesis).
  - Pre-stress injection: optionally expose the agent to threat/failure events
    *before* the task begins, simulating ecologically valid prior stress.
  - Stronger failure appraisals: wrong predictions lower perceived controllability
    and generate uncontrolled-failure-grade affective updates.
  - Multi-episode affect carry-over: ``run_block()`` runs consecutive episodes
    while the agent's affect state accumulates across them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Sequence

from .affect import ConstitutiveAffectSystem
from .benchmark import CandidateRule, CausalMicroWorld, consistent_rules, default_rules
from .events import EVENTS, apply_event_to_homeostasis
from .models import AffectState, Appraisal, CognitiveModulation, HomeostaticState, clamp
from .modulation import derive_modulation


# ---------------------------------------------------------------------------
# Pre-stress protocol: inject a sequence of aversive events before the task
# ---------------------------------------------------------------------------

#: Default stress induction sequence — models moderate ecological pre-stress
#: (e.g. time pressure, prior failures, resource scarcity).
DEFAULT_PRE_STRESS: tuple[str, ...] = (
    "threat",
    "uncontrolled_failure",
    "conflict",
)


def inject_pre_stress(
    affect: ConstitutiveAffectSystem,
    homeostasis: HomeostaticState,
    event_names: Sequence[str] = DEFAULT_PRE_STRESS,
) -> HomeostaticState:
    """Push affect system through a sequence of pre-task stressors.

    Returns the updated homeostatic state; the affect system is mutated
    in-place (by reference).
    """
    for name in event_names:
        event = EVENTS[name]
        homeostasis = apply_event_to_homeostasis(homeostasis, event)
        affect.step(event.appraisal, homeostasis, cognitive_cost=0.45)
    return homeostasis


@dataclass(frozen=True, slots=True)
class StoredObservation:
    point: tuple[int, int]
    outcome: bool
    salience: float
    step: int


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
    working_memory_capacity: int
    forgotten_observations: int
    perseveration_count: int
    strategy_switches: int
    final_arousal: float
    final_stress: float
    final_frustration: float
    pre_stress: bool = False
    block_position: int = 0


class HypothesisAgent:
    """Causal-discovery agent with cognitive modulation and working memory gating."""

    MODES = {"stateless", "generic", "affective", "affective_lesioned"}

    def __init__(
        self,
        mode: str,
        seed: int,
        rules: Sequence[CandidateRule] | None = None,
        *,
        affect_state: AffectState | None = None,
        homeostasis: HomeostaticState | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(self.MODES)}")
        self.mode = mode
        self.rng = random.Random(seed)
        self.rules = list(rules if rules is not None else default_rules())
        self.memory: list[StoredObservation] = []
        self.all_observations: list[tuple[tuple[int, int], bool]] = []
        self.affect = ConstitutiveAffectSystem(state=affect_state)
        self.generic_state = 0.2
        self.homeostasis = homeostasis if homeostasis is not None else HomeostaticState()
        self.last_appraisal = Appraisal()
        self.active_hypothesis: CandidateRule | None = None
        self.perseveration_count = 0
        self.strategy_switches = 0
        self.forgotten_count = 0
        self.step_counter = 0

    @property
    def observations(self) -> list[tuple[tuple[int, int], bool]]:
        """Return raw observations currently retained in active working memory."""
        return [(item.point, item.outcome) for item in self.memory]

    def candidates(self) -> list[CandidateRule]:
        """Rules consistent with active working memory."""
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
        if self.mode == "affective_lesioned":
            # Lesion ablation: working memory and plasticity are protected from stress degradation
            raw = derive_modulation(self.affect.state, self.last_appraisal)
            return replace(
                raw,
                working_memory_gain=0.85,
                plasticity_gain=0.85,
            )
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
        # Stateless baseline
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

    def current_capacity(self) -> int:
        """Compute working memory buffer capacity in items [2, 8]."""
        if self.mode == "stateless":
            return 3  # strictly limited working memory
        if self.mode == "generic":
            return 5  # constant medium memory
        if self.mode == "affective_lesioned":
            return 7  # protected working memory
        # In affective mode, working memory is degraded by acute stress and excessive arousal
        mod = self._modulation()
        return max(2, min(8, round(2.0 + 6.0 * mod.working_memory_gain)))

    def choose_experiment(self, world: CausalMicroWorld) -> tuple[int, int]:
        tried = {obs[0] for obs in self.observations}
        available = [point for point in world.domain if point not in tried]
        if not available:
            return self.rng.choice(world.domain)

        candidates = self.candidates() or self.rules
        modulation = self._modulation()
        temperature = 0.05 + 0.85 * modulation.exploration

        # Score available points by expected information gain + creative divergence
        scores = []
        for point in available:
            information = self._information_score(point, candidates)
            boundary_novelty = (abs(point[0] - point[1]) / 5.0) * modulation.creative_divergence
            scores.append(information + 0.18 * boundary_novelty)

        maximum = max(scores)
        weights = [math.exp((score - maximum) / temperature) for score in scores]
        return self.rng.choices(available, weights=weights, k=1)[0]

    def observe(self, point: tuple[int, int], outcome: bool) -> None:
        self.step_counter += 1
        before = self.candidates() or self.rules
        predicted_probability = sum(rule(*point) for rule in before) / len(before)
        prediction_error = float(outcome) - predicted_probability
        self.all_observations.append((point, outcome))

        # Check if active hypothesis was disconfirmed
        modulation = self._modulation()
        hypothesis_wrong = (
            self.active_hypothesis is not None
            and self.active_hypothesis(*point) != outcome
        )
        if hypothesis_wrong:
            # Disconfirmation occurred. Switching depends on cognitive plasticity.
            # Low plasticity (induced by prolonged stress) produces perseveration.
            if self.rng.random() > modulation.plasticity_gain:
                self.perseveration_count += 1
            else:
                self.active_hypothesis = None
                self.strategy_switches += 1

        # Calculate salience of the observation (higher prediction error = higher salience)
        salience = clamp(
            abs(prediction_error) * modulation.memory_write_gain
            + 0.25 * modulation.attention_gain,
            0.05,
            1.0,
        )
        new_obs = StoredObservation(
            point=point,
            outcome=outcome,
            salience=salience,
            step=self.step_counter,
        )
        self.memory.append(new_obs)

        # Apply working memory capacity constraint
        capacity = self.current_capacity()
        if len(self.memory) > capacity:
            # Evict the item with the lowest salience (salience-gated memory consolidation)
            min_idx = min(range(len(self.memory)), key=lambda i: self.memory[i].salience)
            self.memory.pop(min_idx)
            self.forgotten_count += 1

        after = self.candidates()
        information_gain = clamp((len(before) - len(after)) / max(1, len(before)), 0.0, 1.0)
        certainty = 1.0 - min(1.0, max(0, len(after) - 1) / max(1, len(self.rules) - 1))
        novelty = 1.0 if self.step_counter == 1 else clamp(abs(prediction_error), 0.0, 1.0)

        # Stronger failure appraisal when prediction was confidently wrong:
        # lower perceived controllability, higher urgency — drives stress accumulation.
        if hypothesis_wrong:
            goal_congruence = clamp(-0.65 - 0.35 * abs(prediction_error), -1.0, 1.0)
            controllability = clamp(0.30 - 0.20 * abs(prediction_error), 0.0, 1.0)
            urgency = clamp(0.55 + 0.35 * (self.step_counter / 8.0), 0.0, 1.0)
        else:
            goal_congruence = clamp(2.0 * information_gain - 0.25, -1.0, 1.0)
            controllability = 0.85
            urgency = clamp(self.step_counter / 8.0, 0.0, 1.0)

        self.last_appraisal = Appraisal(
            novelty=novelty,
            goal_congruence=goal_congruence,
            controllability=controllability,
            certainty=certainty,
            agency=0.8 if not hypothesis_wrong else 0.2,
            urgency=urgency,
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

        if self.mode in {"affective", "affective_lesioned"}:
            cognitive_cost = 0.55 if hypothesis_wrong else 0.35
            self.affect.step(self.last_appraisal, self.homeostasis, cognitive_cost=cognitive_cost)
        elif self.mode == "generic":
            self.generic_state = 0.78 * self.generic_state + 0.22 * abs(prediction_error)

        # Update active hypothesis if none currently held
        if self.active_hypothesis is None and after:
            self.active_hypothesis = after[0]

    def select_rule(self) -> CandidateRule:
        candidates = self.candidates()
        if not candidates:
            return self.rng.choice(self.rules)
        modulation = self._modulation()
        # High verification selects top-ranked rule; low verification may pick randomly
        if len(candidates) == 1 or self.rng.random() < modulation.verification:
            return candidates[0]
        return self.rng.choice(candidates)


def run_episode(
    mode: str,
    seed: int,
    budget: int = 6,
    difficulty: str = "ambiguous",
    world: CausalMicroWorld | None = None,
    *,
    pre_stress: bool = False,
    pre_stress_events: Sequence[str] | None = None,
    affect_state: AffectState | None = None,
    homeostasis: HomeostaticState | None = None,
    block_position: int = 0,
) -> EpisodeResult:
    """Run a single causal-discovery episode.

    Parameters
    ----------
    pre_stress : bool
        If True, inject aversive events before the task begins.
    pre_stress_events : sequence of str, optional
        Custom event names for pre-stress induction (default: DEFAULT_PRE_STRESS).
    affect_state : AffectState, optional
        Carry-over affect from a previous episode (multi-episode blocks).
    homeostasis : HomeostaticState, optional
        Carry-over homeostatic state from a previous episode.
    block_position : int
        Position of this episode within a multi-episode block (0-indexed).
    """
    if world is None:
        world = CausalMicroWorld.generate(seed, difficulty=difficulty)
    agent = HypothesisAgent(
        mode=mode,
        seed=seed + 10_000,
        rules=world.rules,
        affect_state=affect_state,
        homeostasis=homeostasis,
    )

    # Pre-stress induction: inject threat/failure events before the task
    if pre_stress and mode in {"affective", "affective_lesioned"}:
        events = pre_stress_events if pre_stress_events is not None else DEFAULT_PRE_STRESS
        agent.homeostasis = inject_pre_stress(agent.affect, agent.homeostasis, events)

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
        experiments=len(agent.all_observations),
        remaining_hypotheses=len(agent.candidates()),
        working_memory_capacity=agent.current_capacity(),
        forgotten_observations=agent.forgotten_count,
        perseveration_count=agent.perseveration_count,
        strategy_switches=agent.strategy_switches,
        final_arousal=state.arousal if "affective" in mode else 0.0,
        final_stress=state.stress if "affective" in mode else 0.0,
        final_frustration=state.frustration if "affective" in mode else 0.0,
        pre_stress=pre_stress,
        block_position=block_position,
    )


def run_block(
    mode: str,
    seed: int,
    block_size: int = 4,
    budget: int = 6,
    difficulty: str = "ambiguous",
    *,
    pre_stress: bool = False,
    pre_stress_events: Sequence[str] | None = None,
) -> list[EpisodeResult]:
    """Run a block of consecutive episodes with affect carry-over.

    The agent's affect and homeostatic state persist across episodes within
    the block, modelling cumulative stress accumulation. Working memory and
    observations are reset per episode (new world), but the emotional state
    carries over — exactly as in real experimental psychology fatigue blocks.

    Returns one EpisodeResult per episode in the block.
    """
    results: list[EpisodeResult] = []
    carry_affect: AffectState | None = None
    carry_homeo: HomeostaticState | None = None

    for position in range(block_size):
        world_seed = seed + position
        world = CausalMicroWorld.generate(world_seed, difficulty=difficulty)
        result = run_episode(
            mode=mode,
            seed=world_seed,
            budget=budget,
            difficulty=difficulty,
            world=world,
            pre_stress=pre_stress and position == 0,  # only pre-stress on first episode
            pre_stress_events=pre_stress_events,
            affect_state=carry_affect,
            homeostasis=carry_homeo,
            block_position=position,
        )
        results.append(result)

        # Carry over affect state for next episode in the block
        # (reconstruct agent briefly to get final state)
        if mode in {"affective", "affective_lesioned"}:
            # Build a temporary agent with same state to extract carry-over
            tmp = HypothesisAgent(
                mode=mode,
                seed=world_seed + 10_000,
                rules=world.rules,
                affect_state=carry_affect,
                homeostasis=carry_homeo,
            )
            if pre_stress and position == 0:
                tmp.homeostasis = inject_pre_stress(
                    tmp.affect, tmp.homeostasis,
                    pre_stress_events or DEFAULT_PRE_STRESS,
                )
            for _ in range(budget):
                if len(tmp.candidates()) == 1:
                    break
                point = tmp.choose_experiment(world)
                tmp.observe(point, world.experiment(point))
            carry_affect = tmp.affect.state
            carry_homeo = tmp.homeostasis

    return results
