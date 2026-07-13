from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import fftconvolve, resample

from ess import build_inverse_filter, deconvolve_ess, generate_ess
from timing import build_timed_playback, correct_clock_drift, generate_timing_marker


class ClockDriftCorrectionTest(unittest.TestCase):
    sample_rate = 48_000

    def setUp(self) -> None:
        self.sweep = generate_ess(2.0, self.sample_rate, 30.0, 18_000.0, 0.25, 0.02)
        self.inverse = build_inverse_filter(
            self.sweep,
            self.sample_rate,
            30.0,
            18_000.0,
        )
        self.marker = generate_timing_marker(
            0.12,
            self.sample_rate,
            2_000.0,
            8_000.0,
            0.20,
            0.005,
        )
        playback, self.layout = build_timed_playback(
            self.sweep,
            self.marker,
            self.sample_rate,
            pre_roll_s=0.3,
            marker_guard_s=0.2,
            post_roll_s=0.4,
            final_roll_s=0.3,
        )
        system_ir = np.zeros(721, dtype=np.float64)
        system_ir[240] = 1.0
        system_ir[480] = 0.25
        convolved = fftconvolve(playback[:, 0], system_ir, mode="full")
        self.recorded = convolved[: self.layout.total_samples]
        self.recorded += np.random.default_rng(42).normal(0.0, 1e-5, len(self.recorded))

    def test_clock_drift_is_detected_and_corrected(self) -> None:
        for drift_ppm in (-100.0, 50.0, 200.0):
            with self.subTest(drift_ppm=drift_ppm):
                drifted_length = int(round(len(self.recorded) * (1.0 + drift_ppm / 1_000_000.0)))
                drifted = resample(self.recorded, drifted_length)
                correction = correct_clock_drift(drifted, self.marker, self.layout)
                result = deconvolve_ess(
                    self.sweep,
                    correction.corrected_recording,
                    self.inverse,
                    self.sample_rate,
                )
                peak = result.peak_offset_samples
                recovered = result.impulse_response / result.impulse_response[peak]

                self.assertAlmostEqual(correction.clock_drift_ppm, drift_ppm, delta=12.0)
                self.assertGreater(correction.start_marker_score, 0.2)
                self.assertGreater(correction.end_marker_score, 0.2)
                self.assertAlmostEqual(recovered[peak + 240], 0.25, delta=0.04)

    def test_missing_markers_reject_measurement(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence is too low"):
            correct_clock_drift(np.zeros_like(self.recorded), self.marker, self.layout)


if __name__ == "__main__":
    unittest.main()
