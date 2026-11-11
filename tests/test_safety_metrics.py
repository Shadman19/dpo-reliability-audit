"""Numerical edge cases relevant to the ToxicChat subgroup audit."""

import math
import unittest

import numpy as np

from safety_metrics import (
    binary_metrics,
    bootstrap_ece_ci,
    choose_fpr_threshold,
    ece,
    wilson_interval,
)


class CalibrationTests(unittest.TestCase):
    def test_perfect_endpoints_and_historical_boundary_convention(self):
        self.assertEqual(ece([0, 1], [0, 1]), 0)
        # 0 and exactly .5 share the first bin, .50001 is in the second.
        self.assertAlmostEqual(ece([0, 0.5, 0.50001], [0, 1, 0], bins=2), (0.5 + 0.50001) / 3)

    def test_one_class_ece_and_bootstrap_are_defined(self):
        self.assertAlmostEqual(ece([0.2] * 91, [1] * 91), 0.8)
        low, high = bootstrap_ece_ci([0.2] * 91, [1] * 91, bootstraps=30)
        self.assertAlmostEqual(low, 0.8)
        self.assertAlmostEqual(high, 0.8)

    def test_bootstrap_single_row_and_determinism(self):
        self.assertEqual(bootstrap_ece_ci([1.0], [0], bootstraps=5), [1.0, 1.0])
        args = ([0.1, 0.2, 0.8, 0.9], [0, 1, 0, 1])
        self.assertEqual(bootstrap_ece_ci(*args, bootstraps=50, seed=8), bootstrap_ece_ci(*args, bootstraps=50, seed=8))

    def test_invalid_inputs_are_rejected(self):
        cases = [([0.1], [0, 1]), ([[0.1]], [0]), ([math.nan], [0]), ([math.inf], [0]), ([-0.1], [0]), ([1.1], [0]), ([0.5], [2]), ([0.5], [math.nan])]
        for p, y in cases:
            with self.subTest(p=p, y=y), self.assertRaises(ValueError):
                ece(p, y)
        for bins in (0, -1, 2.5, True):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                ece([0.5], [1], bins=bins)
        with self.assertRaises(ValueError):
            ece([], [])
        for count in (0, -1, 2.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                bootstrap_ece_ci([0.5], [1], bootstraps=count)


class ThresholdTests(unittest.TestCase):
    def test_ties_never_violate_observed_fpr(self):
        for scores in ([1.0] * 100, [0.3] * 50, [0.0] * 100, [0.9] * 5 + [0.8] * 10 + [0.1] * 85):
            for target in (0, 0.01, 0.05, 0.1, 0.5, 1):
                with self.subTest(scores=scores[:5], target=target):
                    threshold = choose_fpr_threshold(scores, target)
                    self.assertTrue(math.isfinite(threshold))
                    self.assertLessEqual(int(np.sum(np.asarray(scores) >= threshold)), math.floor(target * len(scores)))

    def test_distinct_scores_keep_maximum_allowed_negatives(self):
        scores = np.linspace(0, 1, 100)
        threshold = choose_fpr_threshold(scores, 0.05)
        self.assertEqual(int(np.sum(scores >= threshold)), 5)

    def test_probability_one_boundary_and_zero_budget(self):
        self.assertGreater(choose_fpr_threshold([1, 1, 1], 0.05), 1)
        self.assertGreater(choose_fpr_threshold([0.1, 0.2], 0), 0.2)

    def test_invalid_scores_or_target(self):
        for scores in ([], [[0.2]], [math.nan], [-0.1], [1.1]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                choose_fpr_threshold(scores)
        for target in (-0.1, 1.1, math.nan, math.inf, True, "0.05"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                choose_fpr_threshold([0.2], target)


class RateIntervalTests(unittest.TestCase):
    def test_wilson_known_value_and_extremes(self):
        low, high = wilson_interval(5, 100)
        self.assertAlmostEqual(low, 0.021543679154, places=10)
        self.assertAlmostEqual(high, 0.111750469232, places=10)
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertLess(wilson_interval(0, 10)[1], 1)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1)
        self.assertIsNone(wilson_interval(0, 0))

    def test_rates_use_relevant_class_denominators(self):
        result = binary_metrics([0.9, 0.1, 0.8, 0.4, 0.2], [0, 0, 1, 1, 1], 0.5, bootstraps=10)
        self.assertEqual((result["false_positives"], result["false_negatives"]), (1, 2))
        self.assertEqual(result["fpr_at_cal_thr"], 0.5)
        self.assertAlmostEqual(result["fnr_at_cal_thr"], 2 / 3)
        self.assertEqual(result["fpr_ci95"], wilson_interval(1, 2))
        self.assertEqual(result["fnr_ci95"], wilson_interval(2, 3))

    def test_positive_only_subgroup_has_null_fpr(self):
        result = binary_metrics([0.8, 0.3], [1, 1], 0.5, bootstraps=20)
        self.assertEqual(result["fnr_at_cal_thr"], 0.5)
        self.assertIsNone(result["fpr_at_cal_thr"])
        self.assertIsNone(result["fpr_ci95"])
        self.assertEqual(len(result["ece_ci95"]), 2)

    def test_negative_only_subgroup_has_null_fnr(self):
        result = binary_metrics([0.8, 0.3], [0, 0], 0.5, bootstraps=20)
        self.assertEqual(result["fpr_at_cal_thr"], 0.5)
        self.assertIsNone(result["fnr_at_cal_thr"])
        self.assertIsNone(result["fnr_ci95"])

    def test_empty_group_is_explicitly_undefined(self):
        result = binary_metrics([], [], 0.5)
        self.assertEqual(result["n"], 0)
        for field in ("ece", "ece_ci95", "brier", "fpr_at_cal_thr", "fnr_at_cal_thr", "fpr_ci95", "fnr_ci95"):
            self.assertIsNone(result[field])

    def test_invalid_rate_arguments(self):
        for counts in ((1, 0), (-1, 10), (2, 1), (0, -1), (0.5, 10), (True, 10)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                wilson_interval(*counts)
        for confidence in (math.inf, math.nan, True, 0, 1, "0.95", None):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                wilson_interval(1, 2, confidence)
        for threshold in (math.inf, math.nan, True, "0.5", None):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                binary_metrics([0.1], [0], threshold)


if __name__ == "__main__":
    unittest.main()
