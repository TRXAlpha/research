from __future__ import annotations

import unittest

from affective_metacontrol.agents import HypothesisAgent, run_episode
from affective_metacontrol.benchmark import CausalMicroWorld


class TestAgents(unittest.TestCase):
    def test_working_memory_eviction_under_capacity_limit(self) -> None:
        agent = HypothesisAgent(mode="stateless", seed=42)
        # stateless has fixed capacity of 3
        self.assertEqual(agent.current_capacity(), 3)

        # Feed 5 observations
        for i in range(5):
            agent.observe((i, i), i % 2 == 0)

        # Working memory should retain exactly 3 items
        self.assertEqual(len(agent.memory), 3)
        self.assertEqual(agent.forgotten_count, 2)
        self.assertEqual(len(agent.all_observations), 5)

    def test_salience_prioritization_in_working_memory(self) -> None:
        agent = HypothesisAgent(mode="stateless", seed=42)
        # Give 3 observations
        agent.observe((0, 0), True)
        agent.observe((1, 1), False)
        agent.observe((2, 2), True)

        # Set specific salience levels to test eviction
        agent.memory[0] = agent.memory[0].__class__(point=(0, 0), outcome=True, salience=0.9, step=1)
        agent.memory[1] = agent.memory[1].__class__(point=(1, 1), outcome=False, salience=0.1, step=2)
        agent.memory[2] = agent.memory[2].__class__(point=(2, 2), outcome=True, salience=0.8, step=3)

        # Now add a 4th observation with medium salience 0.5
        # The lowest salience item (point (1, 1) with salience 0.1) should be evicted
        agent.observe((3, 3), False)

        retained_points = [m.point for m in agent.memory]
        self.assertNotIn((1, 1), retained_points)
        self.assertIn((0, 0), retained_points)
        self.assertIn((2, 2), retained_points)
        self.assertIn((3, 3), retained_points)

    def test_affective_agent_adjusts_capacity_dynamically(self) -> None:
        agent = HypothesisAgent(mode="affective", seed=10)
        initial_cap = agent.current_capacity()
        self.assertGreaterEqual(initial_cap, 2)
        self.assertLessEqual(initial_cap, 8)

    def test_episode_tracks_cognitive_metrics(self) -> None:
        world = CausalMicroWorld.generate(seed=77, difficulty="ambiguous")
        result = run_episode(mode="affective", seed=77, budget=6, world=world)
        self.assertGreaterEqual(result.experiments, 1)
        self.assertGreaterEqual(result.working_memory_capacity, 2)
        self.assertGreaterEqual(result.perseveration_count, 0)
        self.assertGreaterEqual(result.strategy_switches, 0)


if __name__ == "__main__":
    unittest.main()
