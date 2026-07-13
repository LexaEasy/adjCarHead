from __future__ import annotations

import unittest

import numpy as np

from measurement_quality import assess_measurement


class MeasurementQualityTest(unittest.TestCase):
    sample_rate = 48_000

    def test_clean_signal_has_band_snr_and_is_accepted(self) -> None:
        rng = np.random.default_rng(42)
        signal = rng.normal(0.0, 0.05, self.sample_rate * 2)
        noise = rng.normal(0.0, 0.001, self.sample_rate // 2)
        quality = assess_measurement(signal, self.sample_rate, signal, noise)

        self.assertTrue(quality["accepted"])
        self.assertEqual(len(quality["band_snr_db"]), 31)
        self.assertGreater(sum(quality["band_confidence_weight"]), 20)
        self.assertGreater(quality["crest_factor_db"], 0)
        self.assertEqual(quality["recommended_peak_window_dbfs"], [-12.0, -6.0])
        self.assertFalse(quality["analog_overload_separable"])

    def test_clipping_and_dropout_are_hard_failures(self) -> None:
        clipped = np.ones(self.sample_rate)
        clipped_quality = assess_measurement(clipped, self.sample_rate)
        self.assertFalse(clipped_quality["accepted"])
        self.assertIn("clipping", clipped_quality["hard_failures"])

        rng = np.random.default_rng(7)
        dropped = rng.normal(0.0, 0.05, self.sample_rate * 2)
        dropped[self.sample_rate // 2 : self.sample_rate] = 0.0
        dropout_quality = assess_measurement(dropped, self.sample_rate)
        self.assertIn("dropout", dropout_quality["hard_failures"])

    def test_pre_and_post_roll_are_excluded_but_active_dropout_is_detected(self) -> None:
        time_s = np.arange(5 * self.sample_rate) / self.sample_rate
        active = 0.1 * np.sin(2.0 * np.pi * 500.0 * time_s)
        full = np.concatenate((np.zeros(self.sample_rate // 10), active, np.zeros(self.sample_rate)))
        quality = assess_measurement(full, self.sample_rate, signal_segment=active)
        self.assertNotIn("dropout", quality["hard_failures"])

        dropped = active.copy()
        dropped[self.sample_rate : 2 * self.sample_rate] = 0.0
        dropout_quality = assess_measurement(full, self.sample_rate, signal_segment=dropped)
        self.assertIn("dropout", dropout_quality["hard_failures"])


if __name__ == "__main__":
    unittest.main()
