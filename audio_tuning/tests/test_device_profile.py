from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from device_profile import LEGACY_PROFILE_SCHEMA_VERSION, load_device_profile


ROOT = Path(__file__).resolve().parents[1]


class DeviceProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_device_profile(ROOT / "devices" / "station2.json")

    def test_profile_exposes_two_measurement_durations(self) -> None:
        self.assertEqual(self.profile.quick_duration_s, 5.0)
        self.assertEqual(self.profile.full_duration_s, 8.0)
        self.assertEqual(len(self.profile.dsp_controls), 5)
        self.assertTrue(self.profile.delays["managed_externally"])
        self.assertEqual(self.profile.schema_version, "car_audio_system_v2")
        self.assertEqual(self.profile.microphone_profile.profile_id, "oklick_sm_700g")
        self.assertEqual(self.profile.microphone_profile.target_optimization_range_hz, (100.0, 18000.0))

    def test_eq_uses_profile_controls_and_point_one_db_steps(self) -> None:
        settings = self.profile.parse_eq("60=1.2,14000=-0.7")
        self.assertEqual(settings["60"], 1.2)
        self.assertEqual(settings["14000"], -0.7)
        with self.assertRaisesRegex(ValueError, "Unknown DSP control"):
            self.profile.parse_eq("999=1.0")
        with self.assertRaisesRegex(ValueError, "0.1 dB steps"):
            self.profile.parse_eq("60=0.15")

    def test_uncharacterized_aura_profile_allows_measurement_but_blocks_dsp(self) -> None:
        profile = load_device_profile(ROOT / "devices" / "aura_indigo_877dsp_mkii.json")

        self.assertFalse(profile.has_subwoofer)
        self.assertTrue(profile.volume_reference_ready)
        self.assertEqual(profile.default_eq(), {})
        self.assertFalse(profile.dsp_recommendation_eligible)
        self.assertEqual(profile.dsp_control_model["band_count"], 48)
        self.assertEqual(profile.input_processing_status, "direct_aux_no_oem_stage")
        self.assertTrue(profile.input_path_validated)
        self.assertFalse(profile.phase_alignment_eligible)
        self.assertFalse(profile.active_crossover_change_allowed)
        self.assertEqual(profile.crossover_policy["mode"], "preserve_passive_network")
        with self.assertRaisesRegex(ValueError, "blocked by the documented speaker topology"):
            profile.require_active_crossover_change_allowed()
        with self.assertRaisesRegex(ValueError, "not characterized"):
            profile.parse_eq("20=0.1")

    def test_legacy_v1_profile_remains_readable(self) -> None:
        payload = {
            "schema_version": LEGACY_PROFILE_SCHEMA_VERSION,
            "device_id": "legacy_fixture",
            "name": "Legacy fixture",
            "has_subwoofer": False,
            "target": "warm_driver",
            "sweep": {"start_hz": 40, "end_hz": 18000, "duration_s": {"quick": 5, "full": 8}},
            "volume": {"fixed": True},
            "processing": {},
            "microphone": {"name": "Legacy USB", "calibration_file": None},
            "validation": {},
            "delays": {"managed_externally": True},
            "dsp_controls": [{"id": "1000", "center_hz": 1000, "step_db": 0.5}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            profile = load_device_profile(path)

        self.assertEqual(profile.schema_version, LEGACY_PROFILE_SCHEMA_VERSION)
        self.assertFalse(profile.dsp_recommendation_eligible)
        self.assertEqual(profile.microphone_profile.profile_id, "legacy_legacy_fixture_microphone")


if __name__ == "__main__":
    unittest.main()
