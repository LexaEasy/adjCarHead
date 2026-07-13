from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import ANALYSIS_FREQUENCIES_HZ
from device_profile import load_device_profile
from frequency_bands import ANALYSIS_SCHEMA_VERSION
from spatial_analysis import aggregate_spatial_payloads
from spatial_positions import SPATIAL_POSITION_KEYS
from spatial_reporting import write_spatial_outputs
from targets import target_curve_db
from tests.result_factory import make_ess_result


ROOT = Path(__file__).resolve().parents[1]


class SpatialReportingTest(unittest.TestCase):
    def test_uncharacterized_aura_writes_report_but_blocks_dsp_suggestion(self) -> None:
        profile = load_device_profile(ROOT / "devices" / "aura_indigo_877dsp_mkii.json")
        target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
        payloads = []
        for index, position in enumerate(SPATIAL_POSITION_KEYS, start=1):
            payload = make_ess_result(
                profile,
                target + 5.0,
                mode="full",
                position=position,
                session_id="aura_fixture",
                date_time=f"2026-07-13T18:{index:02d}:00",
                volume=profile.volume_note(),
                measurement_id=f"aura-{position}",
            )
            payload["quality"]["band_snr_db"] = [30.0] * 31
            payloads.append(payload)
        result = aggregate_spatial_payloads(payloads, target, profile.target_name)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_spatial_outputs(out, result, profile)
            response = json.loads((out / "spatial_response.json").read_text(encoding="utf-8"))
            report = (out / "spatial_report.md").read_text(encoding="utf-8")

        self.assertFalse(response["final_eq_eligible"])
        self.assertEqual(response["equipment"]["microphone_profile"]["profile_id"], "oklick_sm_700g")
        self.assertIn("Точная DSP-рекомендация заблокирована", report)


if __name__ == "__main__":
    unittest.main()
