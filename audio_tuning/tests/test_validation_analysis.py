from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import ANALYSIS_FREQUENCIES_HZ
from device_profile import load_device_profile
from dsp_characterization import characterize_dsp
from dsp_matrix import load_dsp_response_matrix
from dsp_model import suggest_dsp
from targets import target_curve_db
from validation_analysis import analyze_level_linearity, analyze_repeatability
from tests.result_factory import make_ess_result


ROOT = Path(__file__).resolve().parents[1]


class ValidationAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_device_profile(ROOT / "devices" / "station2.json")
        self.target = target_curve_db(self.profile.target_name, ANALYSIS_FREQUENCIES_HZ)

    def payload(
        self,
        raw: np.ndarray,
        volume: str = "reference",
        eq: dict[str, float] | None = None,
        measurement_id: str | None = None,
    ) -> dict[str, object]:
        identifier = f"measurement-{volume}-{sum((eq or {}).values()):.2f}-{float(np.mean(raw)):.3f}"
        return make_ess_result(
            self.profile,
            raw,
            volume=volume,
            eq=eq,
            measurement_id=measurement_id or identifier,
        )

    def test_repeatability_and_level_linearity_accept_stable_shapes(self) -> None:
        repeats = [
            self.payload(self.target + 5.0 + delta)
            for delta in (-0.1, 0.0, 0.1)
        ]
        repeatability = analyze_repeatability(repeats, self.profile)
        levels = [
            (rank, self.payload(self.target + offset, volume=f"level_{rank}"))
            for rank, offset in enumerate((0.0, 5.0, 10.0), start=1)
        ]
        linearity = analyze_level_linearity(levels, self.profile)

        self.assertTrue(repeatability["accepted"])
        self.assertTrue(linearity["accepted"])
        self.assertTrue(linearity["level_response_monotonic"])

    def test_dsp_matrix_is_measured_and_used_for_suggestions(self) -> None:
        baseline = self.payload(self.target + 5.0, measurement_id="baseline")
        variants = []
        for index, control in enumerate(self.profile.dsp_controls):
            influence = np.zeros(31)
            influence[min(30, 5 + index * 5)] = 1.0
            variants.append(
                (
                    control.control_id,
                    2.0,
                    self.payload(
                        self.target + 5.0 + 2.0 * influence,
                        eq={**self.profile.default_eq(), control.control_id: 2.0},
                        measurement_id=f"{control.control_id}-plus",
                    ),
                    -2.0,
                    self.payload(
                        self.target + 5.0 - 2.0 * influence,
                        eq={**self.profile.default_eq(), control.control_id: -2.0},
                        measurement_id=f"{control.control_id}-minus",
                    ),
                )
            )
        matrix_payload = characterize_dsp(baseline, variants, self.profile)
        self.assertTrue(matrix_payload["accepted"])

        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "matrix.json"
            matrix_path.write_text(json.dumps(matrix_payload), encoding="utf-8")
            rejected_path = Path(tmp) / "rejected.json"
            rejected = {**matrix_payload, "accepted": False}
            rejected_path.write_text(json.dumps(rejected), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not passed"):
                load_dsp_response_matrix(rejected_path)

            incomplete_path = Path(tmp) / "incomplete.json"
            incomplete = copy.deepcopy(matrix_payload)
            incomplete["controls"].pop()
            incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_dsp_response_matrix(
                    incomplete_path,
                    {control.control_id for control in self.profile.dsp_controls},
                )

            rejected_control_path = Path(tmp) / "rejected_control.json"
            rejected_control = copy.deepcopy(matrix_payload)
            rejected_control["controls"][0]["accepted"] = False
            rejected_control_path.write_text(json.dumps(rejected_control), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "has not passed"):
                load_dsp_response_matrix(rejected_control_path)

            loaded = load_dsp_response_matrix(
                matrix_path,
                {control.control_id for control in self.profile.dsp_controls},
            )
            characterized = replace(
                self.profile,
                validation={
                    "microphone_processing_disabled": True,
                    "repeatability_verified": True,
                    "volume_linearity_verified": True,
                    "dsp_controls_characterized": True,
                },
                dsp_control_model={
                    **self.profile.dsp_control_model,
                    "status": "characterized",
                    "response_matrix_file": str(matrix_path),
                },
            )
            high_target = target_curve_db(characterized.target_name, loaded.frequencies_hz)
            response = high_target - 2.0 * loaded.response_per_db["60"]
            suggestions = suggest_dsp(
                loaded.frequencies_hz,
                response,
                high_target,
                np.ones(len(loaded.frequencies_hz), dtype=bool),
                characterized,
                characterized.default_eq(),
                np.zeros(len(loaded.frequencies_hz)),
                baseline["quality"],
                baseline["measurement"],
            )
            no_improvement = suggest_dsp(
                loaded.frequencies_hz,
                high_target,
                high_target,
                np.ones(len(loaded.frequencies_hz), dtype=bool),
                characterized,
                characterized.default_eq(),
                np.zeros(len(loaded.frequencies_hz)),
                baseline["quality"],
                baseline["measurement"],
            )

        self.assertEqual(set(loaded.response_per_db), set(characterized.default_eq()))
        self.assertEqual(len(loaded.frequencies_hz), 64)
        self.assertEqual(len(suggestions["suggestions"]), 5)
        self.assertTrue(all(abs(item["new"]) <= 0.3 for item in suggestions["suggestions"]))
        self.assertFalse(no_improvement["recommendation_accepted"])
        self.assertIn(
            "insufficient_predicted_score_improvement",
            no_improvement["recommendation_rejection_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
