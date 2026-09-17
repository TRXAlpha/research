"""Procedural causal-discovery micro-worlds for contamination-resistant tests."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable


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


@dataclass(slots=True)
class CausalMicroWorld:
    target: CandidateRule
    domain: tuple[tuple[int, int], ...]

    @classmethod
    def generate(cls, seed: int, size: int = 6) -> "CausalMicroWorld":
        rng = random.Random(seed)
        rules = default_rules()
        domain = tuple((x, y) for x in range(size) for y in range(size))
        return cls(target=rng.choice(rules), domain=domain)

    def experiment(self, point: tuple[int, int]) -> bool:
        if point not in self.domain:
            raise ValueError(f"Point {point!r} is outside the world domain")
        return self.target(*point)

    def accuracy(self, rule: CandidateRule) -> float:
        correct = sum(rule(*point) == self.target(*point) for point in self.domain)
        return correct / len(self.domain)


def consistent_rules(
    observations: Iterable[tuple[tuple[int, int], bool]],
    rules: Iterable[CandidateRule] | None = None,
) -> list[CandidateRule]:
    candidates = list(rules or default_rules())
    for point, outcome in observations:
        candidates = [rule for rule in candidates if rule(*point) == outcome]
    return candidates

