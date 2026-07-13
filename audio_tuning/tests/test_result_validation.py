from __future__ import annotations

from pathlib import Path
import unittest

from config import ANALYSIS_FREQUENCIES_HZ
from device_profile import load_device_profile
from result_validation import validate_ess_result
from targets import target_curve_db
from tests.result_factory import make_ess_result


ROOT = Path(__file__).resolve().parents[1]


class ResultValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_device_profile(ROOT / "devices" / "station2.json")
        target = target_curve_db(self.profile.target_name, ANALYSIS_FREQUENCIES_HZ)
        self.payload = make_ess_result(self.profile, target)

    def test_current_complete_result_is_accepted(self) -> None:
        self.assertTrue(validate_ess_result(self.payload, expected_mode="quick").accepted)

    def test_old_schema_and_missing_clock_correction_are_rejected(self) -> None:
        self.payload["analysis_schema_version"] = "legacy"
        self.payload["clock_drift_compensated"] = False
        report = validate_ess_result(self.payload, expected_mode="quick")
        self.assertFalse(report.accepted)
        self.assertIn("analysis_schema_version", report.errors)
        self.assertIn("clock_drift_compensated", report.errors)

    def test_incomplete_active_ess_is_rejected(self) -> None:
        self.payload["active_ess_complete"] = False
        self.payload["quality"]["accepted"] = False
        self.payload["quality"]["hard_failures"] = ["incomplete_active_ess"]
        report = validate_ess_result(self.payload, expected_mode="quick")
        self.assertIn("active_ess_complete", report.errors)
