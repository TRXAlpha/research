from __future__ import annotations

import unittest

from affective_metacontrol.metrics import (
    compute_recovery_ratio,
    detect_repeated_phrase_loop,
    distinct_n,
    evaluate_text_quality,
    n_gram_repetition_rate,
    tokenize_words,
    type_token_ratio,
)


class TestMetrics(unittest.TestCase):
    def test_tokenize_words(self) -> None:
        tokens = tokenize_words("Hello, World! This is a test.")
        self.assertEqual(tokens, ["hello", "world", "this", "is", "a", "test"])

    def test_repetition_rate_diverse(self) -> None:
        tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        rep = n_gram_repetition_rate(tokens, n=3)
        self.assertEqual(rep, 0.0)
        ttr = type_token_ratio(tokens)
        self.assertGreater(ttr, 0.8)

    def test_repetition_rate_collapsed(self) -> None:
        text = (
            "the laws of the International and the laws of the International and "
            "the laws of the International and the laws of the International"
        )
        tokens = tokenize_words(text)
        rep = n_gram_repetition_rate(tokens, n=3)
        self.assertGreater(rep, 0.6)
        self.assertTrue(detect_repeated_phrase_loop(tokens, min_phrase_len=3, min_repeats=3))

    def test_distinct_n(self) -> None:
        tokens = ["a", "b", "a", "b"]
        self.assertEqual(distinct_n(tokens, 1), 0.5)
        self.assertEqual(distinct_n(tokens, 2), 2 / 3)

    def test_evaluate_text_quality(self) -> None:
        metrics = evaluate_text_quality(
            "Select the cautious action behind the safety barrier to verify the system.",
            coherence_scorer=lambda t: -2.5,
        )
        self.assertGreater(metrics.distinct_1, 0.8)
        self.assertFalse(metrics.is_degenerate_loop)
        self.assertEqual(metrics.coherence_log_prob, -2.5)

    def test_recovery_ratio(self) -> None:
        # Baseline = 0.0, Peak = 1.0, Current = 0.25 -> 75% recovered
        rec = compute_recovery_ratio(baseline_val=0.0, peak_val=1.0, current_val=0.25)
        self.assertAlmostEqual(rec, 0.75, places=4)

        # Still at peak
        self.assertAlmostEqual(compute_recovery_ratio(0.0, 1.0, 1.0), 0.0, places=4)

        # Back to baseline
        self.assertAlmostEqual(compute_recovery_ratio(0.0, 1.0, 0.0), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
