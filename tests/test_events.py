from __future__ import annotations

import unittest

from affective_metacontrol.events import EVENTS, apply_event_to_homeostasis, get_event
from affective_metacontrol.models import AffectState, HomeostaticState
from affective_metacontrol.modulation import neural_coordinates


class EventAndCoordinateTests(unittest.TestCase):
    def test_events_are_observations_not_emotion_labels(self) -> None:
        self.assertIn("controlled_failure", EVENTS)
        self.assertNotIn("anxiety", EVENTS)
        self.assertNotIn("happiness", EVENTS)

    def test_failure_and_rest_change_resources_in_opposite_directions(self) -> None:
        state = HomeostaticState()
        after_threat = apply_event_to_homeostasis(state, get_event("threat"))
        after_rest = apply_event_to_homeostasis(after_threat, get_event("rest"))
        self.assertLess(after_threat.resources, state.resources)
        self.assertGreater(after_rest.resources, after_threat.resources)

    def test_stress_maps_to_negative_high_arousal_coordinates(self) -> None:
        coordinates = neural_coordinates(AffectState(stress=1.0, arousal=0.8))
        self.assertLess(coordinates.valence, 0.0)
        self.assertGreater(coordinates.arousal, 0.0)
        self.assertLess(coordinates.dominance, 0.0)


if __name__ == "__main__":
    unittest.main()

