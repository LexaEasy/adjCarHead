from __future__ import annotations

import unittest

import numpy as np

from analyze_recordings import locate_active_ess


class ActiveEssLocationTest(unittest.TestCase):
    def test_pre_and_post_roll_are_not_part_of_active_ess(self) -> None:
        rng = np.random.default_rng(42)
        source = rng.normal(0.0, 0.1, 48_000)
        recording = np.concatenate((np.zeros(4_800), source, np.zeros(48_000)))
        self.assertEqual(locate_active_ess(recording, source), 4_800)

    def test_short_recording_reports_zero_start_for_incomplete_detection(self) -> None:
        self.assertEqual(locate_active_ess(np.ones(100), np.ones(200)), 0)
