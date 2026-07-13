from __future__ import annotations

import unittest

import numpy as np

from config import (
    ANALYSIS_FREQUENCIES_HZ,
    ESS_END_HZ,
    ESS_START_HZ,
    NOMINAL_FREQUENCIES_HZ,
    TARGET_CONTROL_DB,
    TARGET_CONTROL_FREQUENCIES_HZ,
    target_curve_db,
)
from dsp import third_octave_levels
from frequency_bands import (
    ANALYSIS_SCHEMA_VERSION,
    EXACT_THIRD_OCTAVE_FREQUENCIES_HZ,
    band_power_levels,
    smooth_fractional_octave,
    third_octave_edges,
)
from reporting import build_table


class ThirdOctaveGridTest(unittest.TestCase):
    def test_iso_base_ten_grid_and_edges_are_contiguous(self) -> None:
        self.assertEqual(len(NOMINAL_FREQUENCIES_HZ), 31)
        self.assertEqual(len(ANALYSIS_FREQUENCIES_HZ), 31)
        for index, center_hz in enumerate(EXACT_THIRD_OCTAVE_FREQUENCIES_HZ):
            expected_hz = 1000.0 * 10.0 ** ((index - 17) / 10.0)
            self.assertAlmostEqual(center_hz, expected_hz, places=10)
        for left, right in zip(
            EXACT_THIRD_OCTAVE_FREQUENCIES_HZ,
            EXACT_THIRD_OCTAVE_FREQUENCIES_HZ[1:],
        ):
            self.assertAlmostEqual(third_octave_edges(left)[1], third_octave_edges(right)[0])
        self.assertAlmostEqual(ESS_START_HZ, third_octave_edges(ANALYSIS_FREQUENCIES_HZ[0])[0])
        self.assertAlmostEqual(ESS_END_HZ, third_octave_edges(ANALYSIS_FREQUENCIES_HZ[-1])[1])

    def test_target_control_points_are_preserved(self) -> None:
        interpolated = target_curve_db(TARGET_CONTROL_FREQUENCIES_HZ)
        np.testing.assert_allclose(interpolated, TARGET_CONTROL_DB, atol=1e-12)

    def test_flat_transfer_is_flat_in_bands_and_smoothed_curve(self) -> None:
        frequencies = np.linspace(1.0, 23_999.0, 96_000)
        power = np.ones_like(frequencies)
        band_power = band_power_levels(frequencies, power, reducer="mean")
        np.testing.assert_allclose(band_power, 1.0, atol=1e-12)
        _, smoothed_db = smooth_fractional_octave(frequencies, power)
        np.testing.assert_allclose(smoothed_db, 0.0, atol=1e-12)
        _, amplified_db = smooth_fractional_octave(frequencies, power * 4.0)
        np.testing.assert_allclose(amplified_db, 10.0 * np.log10(4.0), atol=1e-12)

    def test_one_khz_tone_peaks_in_one_khz_band(self) -> None:
        sample_rate = 48_000
        time_s = np.arange(sample_rate * 4, dtype=np.float64) / sample_rate
        tone = np.sin(2.0 * np.pi * 1000.0 * time_s)
        levels = third_octave_levels(tone, sample_rate)
        peak_index = int(np.argmax(levels))
        self.assertEqual(NOMINAL_FREQUENCIES_HZ[peak_index], 1000.0)

    def test_frequency_table_carries_schema_and_both_centers(self) -> None:
        table = build_table({"test": np.zeros(31)}, "test")
        self.assertEqual(set(table["Analysis_Schema"]), {ANALYSIS_SCHEMA_VERSION})
        self.assertEqual(table["Frequency_Hz"].tolist(), NOMINAL_FREQUENCIES_HZ)
        np.testing.assert_allclose(table["Exact_Center_Hz"], ANALYSIS_FREQUENCIES_HZ)


if __name__ == "__main__":
    unittest.main()
