from __future__ import annotations

import unittest

from affective_metacontrol.agents import run_episode
from affective_metacontrol.benchmark import CausalMicroWorld, default_rules


class BenchmarkTests(unittest.TestCase):
    def test_world_generation_is_reproducible(self) -> None:
        first = CausalMicroWorld.generate(42)
        second = CausalMicroWorld.generate(42)
        self.assertEqual(first.target.name, second.target.name)
        self.assertEqual(first.domain, second.domain)

    def test_target_rule_has_perfect_accuracy(self) -> None:
        world = CausalMicroWorld.generate(3)
        self.assertEqual(world.accuracy(world.target), 1.0)

    def test_all_modes_complete_an_episode(self) -> None:
        for mode in ("stateless", "generic", "affective"):
            result = run_episode(mode=mode, seed=9, budget=5)
            self.assertIn(result.selected_rule, {rule.name for rule in default_rules()})
            self.assertGreaterEqual(result.accuracy, 0.0)
            self.assertLessEqual(result.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()

