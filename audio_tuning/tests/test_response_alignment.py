from __future__ import annotations

import unittest

import numpy as np

from config import NOMINAL_FREQUENCIES_HZ, TARGET_DB
from dsp import zone_errors
from response_alignment import align_response_to_target


class ResponseAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frequencies = np.asarray(NOMINAL_FREQUENCIES_HZ)
        self.target = np.asarray(TARGET_DB)

    def test_constant_level_offset_is_removed_from_shape(self) -> None:
        raw = self.target + 6.0
        result = align_response_to_target(raw, self.target)

        self.assertAlmostEqual(result.offset_db, 6.0)
        self.assertAlmostEqual(result.mad_db, 0.0)
        self.assertEqual(result.band_count, 11)
        self.assertEqual(result.outlier_count, 0)
        np.testing.assert_allclose(result.aligned_response_db, self.target, atol=1e-12)

    def test_one_khz_notch_does_not_shift_the_whole_curve(self) -> None:
        raw = self.target.copy()
        raw[self.frequencies == 1000.0] -= 4.0
        result = align_response_to_target(raw, self.target)
        errors = zone_errors(result.aligned_response_db, self.target)

        self.assertAlmostEqual(result.offset_db, 0.0)
        self.assertEqual(result.outlier_count, 1)
        self.assertLess(errors["Full"], 0.2)
        unchanged = self.frequencies != 1000.0
        np.testing.assert_allclose(
            result.aligned_response_db[unchanged],
            self.target[unchanged],
            atol=1e-12,
        )

    def test_multiple_local_outliers_do_not_control_alignment(self) -> None:
        raw = self.target + 2.5
        raw[self.frequencies == 630.0] -= 8.0
        raw[self.frequencies == 2000.0] += 7.0
        result = align_response_to_target(raw, self.target)

        self.assertAlmostEqual(result.offset_db, 2.5)
        self.assertEqual(result.outlier_count, 2)

    def test_invalid_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            align_response_to_target(np.zeros(3), self.target)


if __name__ == "__main__":
    unittest.main()
