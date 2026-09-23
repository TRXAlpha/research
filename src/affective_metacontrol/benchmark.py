"""Procedural causal-discovery micro-worlds with controlled ambiguity."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable, Sequence


RuleFunction = Callable[[int, int], bool]


@dataclass(frozen=True, slots=True)
class CandidateRule:
    name: str
    function: RuleFunction

    def __call__(self, x: int, y: int) -> bool:
        return bool(self.function(x, y))


def default_rules() -> tuple[CandidateRule, ...]:
    return (
        CandidateRule("sum_ge_6", lambda x, y: x + y >= 6),
        CandidateRule("x_gt_y", lambda x, y: x > y),
        CandidateRule("same_parity", lambda x, y: x % 2 == y % 2),
        CandidateRule("xor_high", lambda x, y: (x >= 3) ^ (y >= 3)),
        CandidateRule("gap_ge_3", lambda x, y: abs(x - y) >= 3),
        CandidateRule("product_even", lambda x, y: (x * y) % 2 == 0),
        CandidateRule("max_ge_5", lambda x, y: max(x, y) >= 5),
        CandidateRule("both_nonzero", lambda x, y: x > 0 and y > 0),
    )


def domain_overlap(
    rule1: CandidateRule,
    rule2: CandidateRule,
    domain: Sequence[tuple[int, int]],
) -> float:
    """Compute fraction of points where two candidate rules make identical predictions."""
    if not domain:
        return 1.0
    matches = sum(rule1(*p) == rule2(*p) for p in domain)
    return matches / len(domain)


def procedural_rules(
    seed: int,
    count: int = 16,
    size: int = 6,
    target_rule: CandidateRule | None = None,
) -> tuple[CandidateRule, ...]:
    """Generate a diverse set of parametric candidate rules over a 2D discrete grid.

    If target_rule is given, deliberately inject near-miss distractor rules
    that overlap with target_rule on 75%-90% of the domain points.
    """
    rng = random.Random(seed)
    domain = tuple((x, y) for x in range(size) for y in range(size))
    pool: dict[str, CandidateRule] = {}

    def make_linear(w1: int, w2: int, bias: int) -> CandidateRule:
        return CandidateRule(f"lin_{w1}x_{w2}y_ge_{bias}", lambda x, y, w1=w1, w2=w2, b=bias: w1 * x + w2 * y >= b)

    def make_radial(cx: int, cy: int, r2: int) -> CandidateRule:
        return CandidateRule(f"rad_{cx}_{cy}_r2_{r2}", lambda x, y, cx=cx, cy=cy, r2=r2: (x - cx) ** 2 + (y - cy) ** 2 <= r2)

    def make_box(x1: int, x2: int, y1: int, y2: int) -> CandidateRule:
        return CandidateRule(f"box_{x1}_{x2}_{y1}_{y2}", lambda x, y, x1=x1, x2=x2, y1=y1, y2=y2: x1 <= x <= x2 and y1 <= y <= y2)

    def make_mod(m: int, r: int) -> CandidateRule:
        return CandidateRule(f"mod_{m}_eq_{r}", lambda x, y, m=m, r=r: (x + y) % m == r)

    def make_diff_mod(m: int, r: int) -> CandidateRule:
        return CandidateRule(f"diff_mod_{m}_eq_{r}", lambda x, y, m=m, r=r: abs(x - y) % m == r)

    # 1. Base procedural generators
    candidates: list[CandidateRule] = []

    # Linear hyperplanes
    for w1 in (1, 2):
        for w2 in (1, -1, 2):
            for b in (3, 5, 6, 8):
                candidates.append(make_linear(w1, w2, b))

    # Radial circles
    for cx in (2, 3):
        for cy in (2, 3):
            for r2 in (4, 8, 12):
                candidates.append(make_radial(cx, cy, r2))

    # Bounded regions
    for x1, x2 in ((1, 4), (2, 5), (0, 3)):
        for y1, y2 in ((1, 4), (2, 5), (0, 3)):
            candidates.append(make_box(x1, x2, y1, y2))

    # Modular arithmetic
    for m in (2, 3, 4):
        for r in range(m):
            candidates.append(make_mod(m, r))
            candidates.append(make_diff_mod(m, r))

    # Filter out trivial rules (always True or always False)
    valid_candidates = []
    for r in candidates:
        positives = sum(r(*p) for p in domain)
        if 4 <= positives <= (len(domain) - 4):
            valid_candidates.append(r)

    rng.shuffle(valid_candidates)

    selected: list[CandidateRule] = []
    if target_rule is not None:
        selected.append(target_rule)
        pool[target_rule.name] = target_rule

        # Sort remaining by near-miss ambiguity: overlap between 0.70 and 0.92
        near_misses = [
            r for r in valid_candidates
            if r.name != target_rule.name and 0.70 <= domain_overlap(r, target_rule, domain) <= 0.92
        ]
        rng.shuffle(near_misses)
        for r in near_misses[: count // 2]:
            if r.name not in pool:
                pool[r.name] = r
                selected.append(r)

    for r in valid_candidates:
        if len(selected) >= count:
            break
        if r.name not in pool:
            pool[r.name] = r
            selected.append(r)

    return tuple(selected)


@dataclass(slots=True)
class CausalMicroWorld:
    target: CandidateRule
    domain: tuple[tuple[int, int], ...]
    rules: tuple[CandidateRule, ...]

    @classmethod
    def generate(
        cls,
        seed: int,
        size: int = 6,
        difficulty: str = "ambiguous",
        rule_count: int = 16,
    ) -> "CausalMicroWorld":
        rng = random.Random(seed)
        domain = tuple((x, y) for x in range(size) for y in range(size))

        if difficulty == "toy":
            rules = default_rules()
            target = rng.choice(rules)
            return cls(target=target, domain=domain, rules=rules)

        # Generate seed target from base rules
        base_pool = procedural_rules(seed=seed, count=32, size=size)
        target = rng.choice(base_pool)

        # Generate world rules centered with controlled ambiguity around target
        world_rules = procedural_rules(
            seed=seed + 500,
            count=rule_count,
            size=size,
            target_rule=target,
        )

        return cls(target=target, domain=domain, rules=world_rules)

    def experiment(self, point: tuple[int, int]) -> bool:
        if point not in self.domain:
            raise ValueError(f"Point {point!r} is outside the world domain")
        return self.target(*point)

    def accuracy(self, rule: CandidateRule) -> float:
        correct = sum(rule(*point) == self.target(*point) for point in self.domain)
        return correct / len(self.domain)


def consistent_rules(
    observations: Iterable[tuple[tuple[int, int], bool] | tuple[tuple[int, int], bool, float]],
    rules: Iterable[CandidateRule] | None = None,
) -> list[CandidateRule]:
    """Filter rules consistent with all provided observations.

    Accepts both 2-tuples (point, outcome) and 3-tuples (point, outcome, salience).
    """
    candidates = list(rules or default_rules())
    for item in observations:
        point = item[0]
        outcome = item[1]
        candidates = [rule for rule in candidates if rule(*point) == outcome]
    return candidates
