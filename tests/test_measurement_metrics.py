"""Prevent missing/stale periods from disappearing from reported P95."""
import unittest
from benchmarks.metrics import sampled_age_summary


class MeasurementTests(unittest.TestCase):
    def test_missing_and_stale_samples_share_the_same_percentile_population(self):
        result = sampled_age_summary([100] * 18 + [500], 1)
        self.assertEqual(result["latest_age_p95_ms"], 500)
        self.assertEqual(result["stale_over_200ms_fraction"], .1)

    def test_more_than_five_percent_missing_has_no_finite_p95(self):
        result = sampled_age_summary([50] * 94, 6)
        self.assertIsNone(result["latest_age_p95_ms"])
        self.assertEqual(result["no_result_fraction"], .06)

    def test_empty_and_invalid_observations_cannot_pass(self):
        self.assertIsNone(sampled_age_summary([], 0)["latest_age_p95_ms"])
        with self.assertRaises(ValueError): sampled_age_summary([-1], 0)


if __name__ == "__main__": unittest.main()
