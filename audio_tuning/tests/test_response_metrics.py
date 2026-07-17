from __future__ import annotations

import unittest

import numpy as np

from response_metrics import calculate_response_error_metrics


class ResponseMetricsTest(unittest.TestCase):
    def test_reports_distribution_and_signed_extremes(self) -> None:
        target = np.zeros(5)
        response = np.asarray([-4.0, -1.0, 0.0, 2.0, 8.0])
        metrics = calculate_response_error_metrics(
            response,
            target,
            np.asarray([True, True, True, True, False]),
        )

        self.assertAlmostEqual(metrics.mean_absolute_error_db, 1.75)
        self.assertAlmostEqual(metrics.median_absolute_error_db, 1.5)
        self.assertAlmostEqual(metrics.percentile_75_absolute_error_db, 2.5)
        self.assertAlmostEqual(metrics.maximum_absolute_error_db, 4.0)
        self.assertEqual(metrics.maximum_positive_deviation_db, 2.0)
        self.assertEqual(metrics.maximum_negative_deviation_db, -4.0)
        self.assertAlmostEqual(metrics.mean_signed_deviation_db, -0.75)
        self.assertEqual(metrics.minimum_response_db, -4.0)
        self.assertEqual(metrics.maximum_response_db, 2.0)
        self.assertEqual(metrics.points_within_3_db, 3)
        self.assertEqual(metrics.band_count, 4)

    def test_rejects_empty_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one finite"):
            calculate_response_error_metrics(
                np.zeros(3), np.zeros(3), np.zeros(3, dtype=bool)
            )


if __name__ == "__main__":
    unittest.main()
