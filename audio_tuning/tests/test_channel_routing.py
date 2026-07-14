from __future__ import annotations

import unittest

import numpy as np

from measure_playrec import route_output


class ChannelRoutingTest(unittest.TestCase):
    def test_left_right_and_stereo_are_distinct(self) -> None:
        signal = np.asarray([0.1, -0.2, 0.3])
        left = route_output(signal, "left", 2)
        right = route_output(signal, "right", 2)
        stereo = route_output(signal, "stereo", 2)

        np.testing.assert_allclose(left[:, 0], signal)
        np.testing.assert_allclose(left[:, 1], 0.0)
        np.testing.assert_allclose(right[:, 0], 0.0)
        np.testing.assert_allclose(right[:, 1], signal)
        np.testing.assert_allclose(stereo[:, 0], signal)
        np.testing.assert_allclose(stereo[:, 1], signal)

    def test_channel_routing_requires_stereo_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-channel"):
            route_output(np.ones(10), "left", 1)
