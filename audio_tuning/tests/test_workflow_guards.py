from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from channel_comparison import write_channel_comparison
from config import ANALYSIS_FREQUENCIES_HZ
from device_profile import load_device_profile
from full_comparison import write_full_comparison
from targets import target_curve_db
from tuning_state import confirm_listening
from tests.result_factory import make_spatial_result


ROOT = Path(__file__).resolve().parents[1]


class WorkflowGuardsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_device_profile(ROOT / "devices" / "station2.json")
        self.target = target_curve_db(self.profile.target_name, ANALYSIS_FREQUENCIES_HZ)

    @staticmethod
    def write(path: Path, payload: dict[str, object]) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_single_full_cannot_be_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(
                Path(tmp) / "candidate.json",
                make_spatial_result(self.profile, self.target, purpose="candidate"),
            )
            with self.assertRaisesRegex(ValueError, "passed full comparison"):
                confirm_listening(path, "operator", "listened")

    def test_full_comparison_requires_listening_before_confirmation(self) -> None:
        baseline_curve = self.target.copy()
        baseline_curve[13:16] += 5.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.write(
                root / "baseline.json",
                make_spatial_result(self.profile, baseline_curve, purpose="baseline"),
            )
            candidate = self.write(
                root / "candidate.json",
                make_spatial_result(self.profile, self.target, purpose="candidate"),
            )
            comparison = write_full_comparison(root, baseline, candidate, self.profile)
            payload = json.loads(comparison.read_text(encoding="utf-8"))
            self.assertEqual(payload["tuning_state"], "listening_confirmation_required")
            self.assertFalse(payload["final_verdict_allowed"])
            confirmed = confirm_listening(comparison, "operator", "balanced on reference tracks")
            confirmed_payload = json.loads(confirmed.read_text(encoding="utf-8"))
            self.assertEqual(confirmed_payload["tuning_state"], "confirmed_preset")

    def test_channel_results_must_not_be_mixed_or_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for channel in ("left", "right", "stereo"):
                payload = make_spatial_result(
                    self.profile,
                    self.target,
                    purpose="baseline",
                    channel=channel,
                    session_id=f"session-{channel}",
                )
                paths.append(self.write(root / f"{channel}.json", payload))
            result = write_channel_comparison(root / "report", paths)
            report = json.loads(result.read_text(encoding="utf-8"))
            self.assertFalse(report["polarity_claim_allowed"])
            self.assertFalse(report["cross_run_delay_claim_allowed"])

            duplicate = json.loads(paths[1].read_text(encoding="utf-8"))
            duplicate["measurement_invariants"]["channel_selection"] = "left"
            paths[1].write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "channel does not match"):
                write_channel_comparison(root / "invalid", paths)
