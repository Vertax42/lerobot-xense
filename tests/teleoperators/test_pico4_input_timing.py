"""Distinguish slow polling, repeated SDK frames, and stationary poses offline."""

import unittest
from lerobot.teleoperators.pico4.input_timing import InputTiming


class TestInputTiming(unittest.TestCase):
    def test_frozen_input_with_healthy_polling(self):
        timing = InputTiming()
        pose = [0, 0, 0, 0, 0, 0, 1]
        timing.observe(0, pose, 100)
        for n in range(1, 51):
            report = timing.observe(n / 100, pose, 100)
        self.assertAlmostEqual(report["poll_hz"], 100)
        self.assertEqual(report["pose_changes"], 0)
        self.assertEqual(report["sdk_stamp_changes"], 0)
        self.assertAlmostEqual(report["sdk_stamp_repeat_ms"], 500)
        self.assertAlmostEqual(report["pose_repeat_ms"], 500)
        self.assertAlmostEqual(report["max_poll_gap_ms"], 10)

    def test_stationary_pose_is_distinct_from_frozen_stream(self):
        timing = InputTiming()
        for n in range(51):
            report = timing.observe(n / 100, [0] * 7, n + 1)
        self.assertEqual(report["pose_changes"], 0)
        self.assertEqual(report["sdk_stamp_changes"], 50)
        self.assertEqual(report["sdk_stamp_repeat_ms"], 0)
        self.assertAlmostEqual(report["pose_repeat_ms"], 500)

    def test_recovery_keeps_the_previous_gap_visible(self):
        timing = InputTiming()
        for n in range(50):
            timing.observe(n / 100, [0] * 7, 100)
        report = timing.observe(0.5, [0.1] * 7, 101)
        self.assertEqual(report["pose_repeat_ms"], 0)
        self.assertEqual(report["sdk_stamp_repeat_ms"], 0)
        self.assertAlmostEqual(report["max_pose_repeat_ms"], 500)
        self.assertAlmostEqual(report["max_sdk_stamp_repeat_ms"], 500)
        for n in range(51, 101):
            report = timing.observe(n / 100, [n * 1e-10] * 7, n + 101)
        self.assertEqual(report["pose_changes"], 50)  # No rounding hides small movements.
        self.assertAlmostEqual(report["max_pose_repeat_ms"], 10)

    def test_missing_timestamp_does_not_report_a_stream_freeze(self):
        for stamp in (None, 0):
            with self.subTest(stamp=stamp):
                timing = InputTiming()
                timing.observe(0, [0] * 7, stamp)
                report = timing.observe(0.5, [0] * 7, stamp)
                self.assertEqual(report["sdk_stamp_repeat_ms"], -1)
                self.assertEqual(report["max_sdk_stamp_repeat_ms"], -1)
                self.assertEqual(report["sdk_stamp_changes"], 0)

    def test_delayed_poll_is_visible_separately(self):
        timing = InputTiming()
        timing.observe(0, [0] * 7, 100)
        report = timing.observe(0.5, [1] * 7, 101)
        self.assertEqual(report["poll_hz"], 2)
        self.assertAlmostEqual(report["max_poll_gap_ms"], 500)

    def test_timestamp_restart_uses_local_monotonic_duration(self):
        timing = InputTiming()
        timing.observe(0, [0] * 7, 1_000_000_000)
        report = timing.observe(0.5, [0] * 7, 1)
        self.assertEqual(report["sdk_stamp_repeat_ms"], 0)
        self.assertEqual(report["sdk_stamp_changes"], 1)



if __name__ == "__main__":
    unittest.main()
