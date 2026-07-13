from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import tempfile
import unittest

import numpy as np

from config import ANALYSIS_FREQUENCIES_HZ, NOMINAL_FREQUENCIES_HZ
from device_profile import load_device_profile
from scoring import score_response
from targets import target_curve_db


ROOT = Path(__file__).resolve().parents[1]


class ScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_device_profile(ROOT / "devices" / "station2.json")
        self.target = target_curve_db(self.profile.target_name, ANALYSIS_FREQUENCIES_HZ)
        self.quality = {"band_confidence_weight": [1.0] * 31}

    def test_observation_and_target_optimization_masks_are_separate(self) -> None:
        score = score_response(self.target, self.target, self.profile, self.quality)
        frequencies = np.asarray(NOMINAL_FREQUENCIES_HZ)
        observed = np.asarray(score.observed_frequency_mask)
        optimized = np.asarray(score.target_optimization_mask)

        self.assertAlmostEqual(score.total_cost, 0.0)
        self.assertFalse(np.any(observed[frequencies < 40.0]))
        self.assertTrue(np.any(observed[(frequencies >= 40.0) & (frequencies < 100.0)]))
        self.assertFalse(np.any(optimized[frequencies < 100.0]))
        self.assertLessEqual(score.observed_high_hz, 18_000.0)
        self.assertEqual(score.optimization_low_hz, 100.0)

    def test_low_confidence_band_is_not_optimized(self) -> None:
        response = self.target.copy()
        response[10] -= 12.0
        weights = [1.0] * 31
        weights[10] = 0.0
        score = score_response(
            response,
            self.target,
            self.profile,
            {"band_confidence_weight": weights},
        )

        self.assertFalse(score.observed_frequency_mask[10])
        self.assertFalse(score.target_optimization_mask[10])
        self.assertAlmostEqual(score.target_error_db, 0.0)

    def test_broad_peak_costs_more_than_a_narrow_deep_null(self) -> None:
        peak_response = self.target.copy()
        peak_response[13:16] += 5.0
        null_response = self.target.copy()
        null_response[14] -= 10.0

        peak_score = score_response(peak_response, self.target, self.profile, self.quality)
        null_score = score_response(null_response, self.target, self.profile, self.quality)

        self.assertGreater(peak_score.target_error_db, null_score.target_error_db)

    def test_only_anomalous_experimental_distortion_is_hard_limited(self) -> None:
        distortion = [1.0] * 31
        distortion[12] = 60.0
        score = score_response(
            self.target,
            self.target,
            self.profile,
            {
                "band_confidence_weight": [1.0] * 31,
                "early_h2_h3_ratio_percent": distortion,
            },
        )

        self.assertTrue(score.observed_frequency_mask[12])
        self.assertFalse(score.target_optimization_mask[12])
        self.assertEqual(score.experimental_distortion_limited_band_count, 1)

    def test_repeatability_artifact_limits_optimization(self) -> None:
        values = [0.1] * 31
        values[12] = 2.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repeatability.json"
            path.write_text(
                json.dumps({"band_standard_deviation_db": values}),
                encoding="utf-8",
            )
            profile = replace(
                self.profile,
                validation_artifacts={"repeatability": str(path)},
            )
            score = score_response(self.target, self.target, profile, self.quality)

        self.assertTrue(score.observed_frequency_mask[12])
        self.assertFalse(score.target_optimization_mask[12])
        self.assertEqual(score.instability_limited_band_count, 1)


if __name__ == "__main__":
    unittest.main()
