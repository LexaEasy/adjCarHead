from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from completion_assessment import assess_completion
from config import ANALYSIS_FREQUENCIES_HZ
from device_profile import load_device_profile
from scoring import score_response
from targets import target_curve_db


ROOT = Path(__file__).resolve().parents[1]


class CompletionAssessmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_device_profile(ROOT / "devices" / "station2.json")
        self.target = target_curve_db(self.profile.target_name, ANALYSIS_FREQUENCIES_HZ)
        self.quality = {
            "band_confidence_weight": [1.0] * 31,
            "worst_peak_dbfs": -12.0,
        }

    def _profile_with_repeatability(self, root: Path) -> object:
        path = root / "repeatability.json"
        path.write_text(
            json.dumps({"band_standard_deviation_db": [0.4] * 31}),
            encoding="utf-8",
        )
        return replace(
            self.profile,
            validation_artifacts={"repeatability": str(path)},
        )

    def test_recommends_stop_only_after_full_evidence_and_plateau(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._profile_with_repeatability(Path(tmp))
            score = score_response(self.target, self.target, profile, self.quality)
            assessment = assess_completion(
                score,
                profile,
                np.full(31, 0.5),
                {"60": -1.0},
                self.quality,
                0.1,
                0.0,
                False,
                "candidate",
            )

        self.assertEqual(assessment.status, "stop_recommended")
        self.assertTrue(assessment.soft_targets_met)
        self.assertTrue(assessment.requires_listening_confirmation)
        self.assertEqual(assessment.phase_check_status, "unsupported_by_current_measurement_path")

    def test_missing_repeatability_prevents_stop(self) -> None:
        score = score_response(self.target, self.target, self.profile, self.quality)
        assessment = assess_completion(
            score,
            self.profile,
            np.full(31, 0.5),
            {},
            self.quality,
            0.1,
            0.0,
            False,
            "baseline",
        )

        self.assertEqual(assessment.status, "insufficient_evidence")
        self.assertIn("repeatability_artifact", assessment.missing_evidence)

    def test_large_remaining_error_requires_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._profile_with_repeatability(Path(tmp))
            response = self.target + 5.0
            score = score_response(response, self.target, profile, self.quality)
            assessment = assess_completion(
                score,
                profile,
                np.full(31, 0.5),
                {},
                self.quality,
                0.1,
                0.0,
                False,
                "candidate",
            )

        self.assertEqual(assessment.status, "continue")
        self.assertIn("mae", assessment.unmet_criteria)

    def test_supported_phase_path_requires_verified_sum_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self._profile_with_repeatability(root)
            phase_path = root / "phase.json"
            phase_path.write_text(
                json.dumps({"summed_level_improvement_db": 0.5}),
                encoding="utf-8",
            )
            profile = replace(
                profile,
                validation_artifacts={
                    **profile.validation_artifacts,
                    "phase_alignment": str(phase_path),
                },
                measurement_reference={
                    "loopback_available": True,
                    "phase_reliable": True,
                },
                speaker_topology={
                    **profile.speaker_topology,
                    "independent_band_channels": True,
                },
            )
            score = score_response(self.target, self.target, profile, self.quality)
            assessment = assess_completion(
                score,
                profile,
                np.full(31, 0.5),
                {},
                self.quality,
                0.1,
                0.0,
                False,
                "candidate",
            )

        self.assertEqual(assessment.status, "continue")
        self.assertEqual(assessment.phase_check_status, "failed")
        self.assertIn("phase_sum_improvement", assessment.unmet_criteria)


if __name__ == "__main__":
    unittest.main()
