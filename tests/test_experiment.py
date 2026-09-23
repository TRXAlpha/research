from __future__ import annotations

import unittest

from affective_metacontrol.experiment import run_study, summarize


class ExperimentTests(unittest.TestCase):
    def test_study_is_balanced_across_modes(self) -> None:
        results = run_study(episodes=4, budget=3, seed=10)
        summary = summarize(results)
        self.assertEqual(set(summary), {"affective", "affective_lesioned", "generic", "stateless"})
        self.assertTrue(all(values["episodes"] == 4 for values in summary.values()))


if __name__ == "__main__":
    unittest.main()

