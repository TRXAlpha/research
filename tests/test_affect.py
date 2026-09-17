from __future__ import annotations

import unittest

from affective_metacontrol.affect import ConstitutiveAffectSystem
from affective_metacontrol.models import AffectState, Appraisal, HomeostaticState
from affective_metacontrol.modulation import derive_modulation


class AffectDynamicsTests(unittest.TestCase):
    def test_state_is_bounded_under_extreme_input(self) -> None:
        system = ConstitutiveAffectSystem()
        appraisal = Appraisal(
            novelty=99,
            goal_congruence=-99,
            controllability=99,
            certainty=-99,
            agency=99,
            urgency=99,
            prediction_error=-99,
        )
        state = system.step(appraisal, HomeostaticState(resources=-99), cognitive_cost=99)
        self.assertGreaterEqual(state.valence, -1.0)
        self.assertLessEqual(state.valence, 1.0)
        for value in (state.arousal, state.stress, state.frustration, state.fatigue, state.mood_arousal):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_affect_persists_but_decays(self) -> None:
        system = ConstitutiveAffectSystem()
        negative = Appraisal(
            novelty=1.0,
            goal_congruence=-1.0,
            controllability=0.1,
            certainty=0.1,
            urgency=1.0,
            prediction_error=-1.0,
        )
        first = system.step(negative, HomeostaticState(resources=0.4, safety=0.2))
        second = system.step(Appraisal(), HomeostaticState())
        self.assertGreater(first.stress, 0.0)
        self.assertGreater(second.stress, 0.0)
        self.assertLess(second.stress, first.stress)

    def test_inverted_u_penalizes_extreme_arousal(self) -> None:
        appraisal = Appraisal(novelty=0.5, certainty=0.5)
        moderate = derive_modulation(AffectState(arousal=0.46), appraisal)
        extreme = derive_modulation(AffectState(arousal=1.0), appraisal)
        self.assertGreater(moderate.cognitive_efficiency, extreme.cognitive_efficiency)
        self.assertGreater(moderate.working_memory_gain, extreme.working_memory_gain)


if __name__ == "__main__":
    unittest.main()

