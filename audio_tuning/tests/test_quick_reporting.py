from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from config import ANALYSIS_FREQUENCIES_HZ
from device_profile import load_device_profile
from quick_reporting import write_quick_outputs
from targets import target_curve_db
from tests.result_factory import make_ess_result


ROOT = Path(__file__).resolve().parents[1]


class QuickReportingTest(unittest.TestCase):
    @staticmethod
    def measurement(profile: object, volume: str = "reference") -> dict[str, object]:
        return {
            "device_profile_id": profile.device_id,
            "device_profile_schema": profile.schema_version,
            "microphone_profile_id": profile.microphone_profile.profile_id,
            "input_device": 1,
            "output_device": 2,
            "sample_rate": 48_000,
            "volume_note": volume,
            "processing_settings": profile.processing,
            "measurement_mode": "quick",
        }

    def test_quick_result_never_claims_final_dsp(self) -> None:
        profile = load_device_profile(ROOT / "devices" / "station2.json")
        target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
        payload = make_ess_result(profile, target + 4.0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ess_result = out / "ess_response.json"
            ess_result.write_text(json.dumps(payload), encoding="utf-8")
            result_path = write_quick_outputs(out, ess_result, profile)
            result = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(result["mode"], "quick")
            self.assertEqual(result["verdict"], "baseline_created")
            self.assertFalse(result["final_dsp_eligible"])
            self.assertTrue((out / "quick_report.md").exists())
            self.assertTrue((out / "quick_comparison.png").exists())

    def test_quick_verdict_has_zone_regression_guards_and_requires_invariants(self) -> None:
        profile = load_device_profile(ROOT / "devices" / "station2.json")
        target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
        baseline_curve = target.copy()
        baseline_curve[13:16] += 5.0
        baseline = make_ess_result(profile, baseline_curve, measurement_id="baseline")
        current = make_ess_result(profile, target, measurement_id="candidate")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            baseline_path = out / "baseline.json"
            current_path = out / "current.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            current_path.write_text(json.dumps(current), encoding="utf-8")
            result_path = write_quick_outputs(out, current_path, profile, baseline_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(result["verdict"], "technically_better_candidate")
            current["measurement"]["volume_note"] = "changed"
            current_path.write_text(json.dumps(current), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "volume_note"):
                write_quick_outputs(out, current_path, profile, baseline_path)

    def test_candidate_cannot_improve_by_losing_reliable_band(self) -> None:
        profile = load_device_profile(ROOT / "devices" / "station2.json")
        target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
        baseline = make_ess_result(profile, target + 3.0, measurement_id="baseline")
        candidate_curve = target.copy()
        candidate_curve[15] -= 30.0
        candidate = make_ess_result(profile, candidate_curve, measurement_id="candidate")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            baseline_path = out / "baseline.json"
            candidate_path = out / "candidate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            result_path = write_quick_outputs(out, candidate_path, profile, baseline_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(result["frequency_coverage_regression"])
        self.assertEqual(result["verdict"], "candidate_rejected_frequency_coverage_regression")


if __name__ == "__main__":
    unittest.main()
