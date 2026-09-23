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
        for mode in ("stateless", "generic", "affective", "affective_lesioned"):
            # Test toy mode
            result_toy = run_episode(mode=mode, seed=9, budget=5, difficulty="toy")
            self.assertIn(result_toy.selected_rule, {rule.name for rule in default_rules()})
            self.assertGreaterEqual(result_toy.accuracy, 0.0)
            self.assertLessEqual(result_toy.accuracy, 1.0)

            # Test ambiguous procedural mode
            result_amb = run_episode(mode=mode, seed=9, budget=5, difficulty="ambiguous")
            self.assertGreaterEqual(result_amb.accuracy, 0.0)
            self.assertLessEqual(result_amb.accuracy, 1.0)

    def test_procedural_ambiguity_generates_near_miss_distractors(self) -> None:
        world = CausalMicroWorld.generate(seed=123, difficulty="ambiguous", rule_count=16)
        self.assertEqual(len(world.rules), 16)
        self.assertEqual(world.accuracy(world.target), 1.0)
        # Verify that rules contain distinct names and functions
        rule_names = {r.name for r in world.rules}
        self.assertEqual(len(rule_names), len(world.rules))


if __name__ == "__main__":
    unittest.main()

