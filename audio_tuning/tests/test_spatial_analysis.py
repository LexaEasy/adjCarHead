from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from config import ANALYSIS_FREQUENCIES_HZ, TARGET_DB
from frequency_bands import ANALYSIS_SCHEMA_VERSION
from device_profile import load_device_profile
from spatial_analysis import aggregate_spatial_payloads
from spatial_positions import SPATIAL_POSITION_KEYS, SPATIAL_POSITIONS, spatial_sequence_text
from tests.result_factory import make_ess_result


ROOT = Path(__file__).resolve().parents[1]


class SpatialAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_device_profile(ROOT / "devices" / "station2.json")

    def make_payload(
        self,
        position: str,
        response_db: np.ndarray,
        *,
        session_id: str = "station2_default",
        volume_note: str = "station=4,laptop=70",
        clipped: bool = False,
        date_time: str = "2026-07-13T16:00:00",
    ) -> dict[str, object]:
        payload = make_ess_result(
            self.profile,
            response_db,
            mode="full",
            position=position,
            session_id=session_id,
            date_time=date_time,
            volume=volume_note,
            measurement_id=f"measurement-{position}",
        )
        payload["quality"]["clipped"] = clipped
        return payload

    def make_complete_session(self) -> list[dict[str, object]]:
        target = np.asarray(TARGET_DB)
        level_offsets = (-1.0, -0.5, 0.0, 0.5, 1.0, 0.0)
        return [
            self.make_payload(
                position,
                target + 5.0 + offset,
                date_time=f"2026-07-13T16:{index:02d}:00",
            )
            for index, (position, offset) in enumerate(
                zip(SPATIAL_POSITION_KEYS, level_offsets),
                start=1,
            )
        ]

    def test_complete_session_is_averaged_in_db_then_aligned(self) -> None:
        result = aggregate_spatial_payloads(self.make_complete_session())

        np.testing.assert_allclose(result.raw_mean_db, np.asarray(TARGET_DB) + 5.0)
        np.testing.assert_allclose(result.aligned_mean_db, TARGET_DB, atol=1e-12)
        self.assertAlmostEqual(result.alignment.offset_db, 5.0)
        self.assertTrue(np.all(result.standard_deviation_db > 0.0))
        self.assertEqual(tuple(result.position_raw_db), SPATIAL_POSITION_KEYS)
        self.assertEqual(result.smoothed_frequencies_hz.shape, (64,))
        self.assertEqual(result.smoothed_aligned_mean_db.shape, (64,))
        self.assertTrue(result.to_dict()["spatial_aggregate_complete"])
        self.assertNotIn("final_eq_eligible", result.to_dict())
        self.assertFalse(result.to_dict()["impulse_responses_averaged"])

    def test_sequence_text_always_shows_full_order_and_current_position(self) -> None:
        text = spatial_sequence_text("front")
        for number, position in enumerate(SPATIAL_POSITIONS, start=1):
            self.assertIn(f"{number}. {position.label}", text)
        self.assertEqual(text.count("ТЕКУЩАЯ ПОЗИЦИЯ"), 1)

    def test_incomplete_or_duplicate_session_is_rejected(self) -> None:
        payloads = self.make_complete_session()
        with self.assertRaisesRegex(ValueError, "Exactly 6"):
            aggregate_spatial_payloads(payloads[:-1])
        payloads[-1]["measurement"]["spatial_position"] = "left_ear"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_spatial_payloads(payloads)
        payloads = self.make_complete_session() + [self.make_complete_session()[0]]
        with self.assertRaisesRegex(ValueError, "Exactly 6"):
            aggregate_spatial_payloads(payloads)

    def test_quick_results_are_rejected_before_aggregation(self) -> None:
        payloads = self.make_complete_session()
        for payload in payloads:
            payload["measurement"]["measurement_mode"] = "quick"
        with self.assertRaisesRegex(ValueError, "measurement_mode"):
            aggregate_spatial_payloads(payloads)

    def test_mismatched_volume_or_clipping_is_rejected(self) -> None:
        payloads = self.make_complete_session()
        payloads[-1]["measurement"]["volume_note"] = "station=5,laptop=70"
        with self.assertRaisesRegex(ValueError, "must match"):
            aggregate_spatial_payloads(payloads)

        payloads = self.make_complete_session()
        payloads[-1]["quality"]["clipped"] = True
        with self.assertRaisesRegex(ValueError, "Clipped"):
            aggregate_spatial_payloads(payloads)

    def test_wrong_recording_order_is_rejected(self) -> None:
        payloads = self.make_complete_session()
        first_time = payloads[0]["measurement"]["date_time"]
        payloads[0]["measurement"]["date_time"] = payloads[1]["measurement"]["date_time"]
        payloads[1]["measurement"]["date_time"] = first_time
        with self.assertRaisesRegex(ValueError, "required order"):
            aggregate_spatial_payloads(payloads)


if __name__ == "__main__":
    unittest.main()
