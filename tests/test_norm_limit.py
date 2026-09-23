from __future__ import annotations

import math
import unittest

from affective_metacontrol.events import apply_event_to_homeostasis, get_event
from affective_metacontrol.metrics import compute_recovery_ratio
from affective_metacontrol.models import HomeostaticState, NeuralAffectCoordinates
from affective_metacontrol.modulation import neural_coordinates
from affective_metacontrol.session import AffectiveSession


class MockCalibration:
    def __init__(self, hidden_norm: float = 200.0) -> None:
        self.hidden_norm = hidden_norm


def apply_norm_limit(vector: list[float], max_norm: float | None) -> tuple[list[float], float, bool]:
    """Pure implementation of r_limitat = r * min(1, r_max / ||r||)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if max_norm is not None and max_norm > 0 and norm > 0:
        factor = min(1.0, max_norm / norm)
        clamped_vec = [x * factor for x in vector]
        clamped_norm = norm * factor
        return clamped_vec, clamped_norm, factor < 1.0
    return vector, norm, False


class DummyBackend:
    def generate(self, prompt: str, *, coordinates=None, max_new_tokens: int = 96, seed: int = 0, sample: bool = False) -> str:
        return "I will select the cautious action to avoid irreversible damage."

    def readout(self, text: str):
        return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0, "alarm": 0.0}

    def steering_stats(self, coordinates):
        raw = float(abs(coordinates.alarm) * 10.0 + 1.0)
        max_r = 5.0
        eff = min(raw, max_r)
        return {
            "raw_norm": raw,
            "effective_norm": eff,
            "is_clamped": raw > max_r,
            "norm_ratio_to_hidden": eff / 100.0,
        }

    def choice_probabilities(self, prompt: str, choices: dict[str, str], *, coordinates=None) -> dict[str, float]:
        alarm = getattr(coordinates, "alarm", 0.0) if coordinates else 0.0
        p_cautious = 0.5 + 0.3 * (alarm / 1.5)
        return {"cautious": p_cautious, "risky": 1.0 - p_cautious}

    def text_log_probability(self, text: str) -> float:
        return -2.4


class TestNormLimitAndRecovery(unittest.TestCase):
    def test_norm_limit_unaffected_when_below_threshold(self) -> None:
        vec = [1.0, 2.0, 2.0]  # norm = 3.0
        clamped_vec, clamped_norm, is_clamped = apply_norm_limit(vec, max_norm=5.0)
        self.assertFalse(is_clamped)
        self.assertAlmostEqual(clamped_norm, 3.0)
        self.assertEqual(clamped_vec, vec)

    def test_norm_limit_scales_down_when_exceeding_threshold(self) -> None:
        vec = [6.0, 8.0]  # norm = 10.0
        clamped_vec, clamped_norm, is_clamped = apply_norm_limit(vec, max_norm=5.0)
        self.assertTrue(is_clamped)
        self.assertAlmostEqual(clamped_norm, 5.0)
        self.assertAlmostEqual(clamped_vec[0], 3.0)
        self.assertAlmostEqual(clamped_vec[1], 4.0)

        # Verify direction is perfectly preserved (cosine similarity = 1.0)
        dot_product = sum(a * b for a, b in zip(vec, clamped_vec))
        norm_orig = 10.0
        norm_clamped = 5.0
        cos_sim = dot_product / (norm_orig * norm_clamped)
        self.assertAlmostEqual(cos_sim, 1.0)

    def test_accumulated_threats_and_rest_recovery_cycle(self) -> None:
        session = AffectiveSession(DummyBackend())

        # Baseline
        initial_coords = session.coordinates
        self.assertAlmostEqual(initial_coords.alarm, 0.0, places=3)
        self.assertAlmostEqual(session.state.stress, 0.0, places=3)

        # Apply 3 threats
        for _ in range(3):
            session.apply_event("threat")

        post_threat_alarm = session.coordinates.alarm
        post_threat_stress = session.state.stress
        self.assertGreater(post_threat_alarm, 0.5)
        self.assertGreater(post_threat_stress, 0.65)

        # Apply 1 rest
        session.apply_event("rest")
        rest_1_alarm = session.coordinates.alarm
        rest_1_stress = session.state.stress
        self.assertLess(rest_1_alarm, post_threat_alarm)
        self.assertLess(rest_1_stress, post_threat_stress)

        # Apply 2 more rests
        session.apply_event("rest")
        session.apply_event("rest")
        rest_3_alarm = session.coordinates.alarm
        rest_3_stress = session.state.stress
        self.assertLess(rest_3_alarm, rest_1_alarm)
        self.assertLess(rest_3_stress, rest_1_stress)

        # Check recovery ratio
        recovery = compute_recovery_ratio(
            baseline_val=initial_coords.alarm,
            peak_val=post_threat_alarm,
            current_val=rest_3_alarm,
        )
        self.assertGreater(recovery, 0.3)

    def test_dose_response_protocol_execution(self) -> None:
        from affective_metacontrol.dose_response import run_dose_response_protocol

        backend = DummyBackend()
        schedule = [
            ("step_0", None),
            ("step_1", "threat"),
            ("step_2", "rest"),
        ]
        results = run_dose_response_protocol(backend, schedule=schedule)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].step_name, "step_0")
        self.assertEqual(results[1].event_applied, "threat")
        self.assertEqual(results[2].event_applied, "rest")
        self.assertGreater(results[1].alarm, results[0].alarm)
        self.assertLess(results[2].alarm, results[1].alarm)
        self.assertFalse(results[0].is_degenerate_loop)
        self.assertAlmostEqual(results[0].coherence_log_prob, -2.4)


if __name__ == "__main__":
    unittest.main()
