"""Distinguish slow polling, repeated SDK frames, and stationary poses offline."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lerobot.teleoperators.pico4.config_pico4 import Pico4Config
from lerobot.teleoperators.pico4.input_timing import InputTiming
from lerobot.teleoperators.pico4.teleop_pico4 import Pico4


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


class TestPicoInputTrace(unittest.TestCase):
    def test_trace_preserves_actions_and_supports_older_sdk(self):
        with TemporaryDirectory() as directory:
            for side in ("left", "right"):
                for has_timestamp in (False, True):
                    with self.subTest(side=side, has_timestamp=has_timestamp):
                        pose = [0.1, 0.2, 0.3, 0, 0, 0, 1]
                        sdk = SimpleNamespace(close=Mock())
                        for name, value in {
                            f"get_{side}_controller_pose": lambda pose=pose: pose,
                            f"get_{side}_grip": lambda: 1.0,
                            f"get_{side}_trigger": lambda: 0.25,
                            "get_A_button": lambda: False,
                            "get_B_button": lambda: False,
                            "get_X_button": lambda: False,
                            "get_Y_button": lambda: False,
                        }.items():
                            setattr(sdk, name, value)
                        if has_timestamp:
                            sdk.get_time_stamp_ns = Mock(return_value=123)
                        controllers = []
                        for trace in (False, True):
                            controller = Pico4(
                                Pico4Config(
                                    calibration_dir=Path(directory),
                                    trace_input=trace,
                                    use_right_controller=side == "right",
                                    use_left_controller=side == "left",
                                    max_pos_velocity=0,
                                    max_rot_velocity=0,
                                )
                            )
                            controller._xrt = sdk
                            controller._is_connected = True
                            controller.logger = Mock()
                            self.addCleanup(controller.disconnect)
                            controllers.append(controller)
                        for now in (0.0, 0.25, 0.5):
                            pose[0] += 0.01
                            with patch("lerobot.teleoperators.pico4.teleop_pico4.time.monotonic", return_value=now):
                                plain, traced = (controller.get_action() for controller in controllers)
                            self.assertEqual(plain, traced)
                        controllers[0].logger.info.assert_not_called()
                        controllers[1].logger.info.assert_called_once()
                        self.assertIn("[PICO_INPUT]", controllers[1].logger.info.call_args.args[0])
                        if has_timestamp:
                            self.assertEqual(sdk.get_time_stamp_ns.call_count, 3)


if __name__ == "__main__":
    unittest.main()
