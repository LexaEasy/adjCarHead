from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import fftconvolve

from dsp import psd_ratio_estimate
from ess import build_inverse_filter, deconvolve_ess, generate_ess


class EssAnalysisTest(unittest.TestCase):
    sample_rate = 48_000
    start_hz = 30.0
    end_hz = 18_000.0

    def make_sweep(self) -> np.ndarray:
        return generate_ess(
            duration_s=2.0,
            sample_rate=self.sample_rate,
            start_hz=self.start_hz,
            end_hz=self.end_hz,
            level=0.25,
            fade_s=0.02,
        )

    def test_deconvolution_recovers_known_impulse_response(self) -> None:
        sweep = self.make_sweep()
        inverse = build_inverse_filter(
            sweep,
            self.sample_rate,
            self.start_hz,
            self.end_hz,
        )
        expected = np.zeros(721, dtype=np.float64)
        expected[0] = 1.0
        expected[240] = 0.35
        expected[480] = -0.20
        pre_roll_samples = 2_400
        recorded = np.pad(
            fftconvolve(sweep, expected, mode="full"),
            (pre_roll_samples, 4_800),
        )
        recorded += np.random.default_rng(42).normal(0.0, 1e-5, len(recorded))

        result = deconvolve_ess(sweep, recorded, inverse, self.sample_rate)
        peak = result.peak_offset_samples
        recovered = result.impulse_response / result.impulse_response[peak]

        self.assertAlmostEqual(result.alignment_delay_s, pre_roll_samples / self.sample_rate, places=4)
        self.assertAlmostEqual(recovered[peak + 240], 0.35, delta=0.02)
        self.assertAlmostEqual(recovered[peak + 480], -0.20, delta=0.02)

    def test_psd_ratio_identity_is_zero_db(self) -> None:
        sweep = self.make_sweep()
        response = psd_ratio_estimate(sweep, sweep, self.sample_rate)
        np.testing.assert_allclose(response, 0.0, atol=1e-10)

    def test_ess_separates_early_second_and_third_harmonics(self) -> None:
        sweep = self.make_sweep()
        inverse = build_inverse_filter(sweep, self.sample_rate, self.start_hz, self.end_hz)
        recorded = sweep + 0.4 * sweep**2 + 0.2 * sweep**3

        result = deconvolve_ess(
            sweep,
            recorded,
            inverse,
            self.sample_rate,
            sweep_start_hz=self.start_hz,
            sweep_end_hz=self.end_hz,
        )

        self.assertIsNotNone(result.distortion_diagnostics)
        values = np.asarray(
            [value for value in result.early_h2_h3_ratio_percent if value is not None]
        )
        self.assertGreater(np.median(values), 4.0)
        self.assertLess(np.median(values), 6.0)
        self.assertEqual(result.distortion_diagnostics["window_duration_s"], 0.12)


if __name__ == "__main__":
    unittest.main()
