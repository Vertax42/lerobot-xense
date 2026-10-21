"""Hardware-free Elite gripper publisher and collision startup regressions."""

import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lerobot.grippers.gripper import GripperSnapshot
from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT


class PublisherTests(unittest.TestCase):
    def robot(self):
        robot = object.__new__(BiEliteCS66RT)
        robot._is_connected = False
        robot._driver = {}
        robot._dashboard = {}
        robot._rtsi = {}
        robot._station_gripper_stop = threading.Event()
        robot._station_gripper_thread = None
        robot._station_gripper_error = None
        robot._coordinator = None
        robot.config = SimpleNamespace(native_collision_gripper_publish_hz=100.0, native_collision_profile=None)
        robot._gripper = {side: Mock() for side in ("left", "right")}
        for gripper in robot._gripper.values():
            gripper.get_gripper_snapshot.return_value = GripperSnapshot(0.2, time.monotonic(), 2, True)
        robot._station = Mock()
        return robot

    def test_publisher_runs_with_held_arms_without_observations_then_joins(self):
        robot = self.robot()
        station = robot._station
        published = threading.Event()
        seen = []

        def publish(samples):
            seen.append(samples)
            if len(seen) >= 3:
                published.set()

        station.publish_gripper_snapshots.side_effect = publish
        robot._start_station_gripper_publisher()
        thread = robot._station_gripper_thread
        self.assertTrue(published.wait(0.5))
        robot._stop_station_layer()
        self.assertFalse(thread.is_alive())
        self.assertIsNone(robot._station)
        self.assertIsNone(robot._station_gripper_thread)
        self.assertGreaterEqual(len(seen), 3)
        for gripper in robot._gripper.values():
            gripper.get_gripper_position.assert_not_called()
            gripper.set_gripper_position.assert_not_called()

    def test_snapshot_exception_invalidates_the_whole_pair(self):
        robot = self.robot()
        station = robot._station
        robot._gripper["left"].get_gripper_snapshot.side_effect = RuntimeError("sensor unavailable")
        station.publish_gripper_snapshots.side_effect = lambda samples: robot._station_gripper_stop.set()
        robot._station_gripper_loop(station)
        station.publish_gripper_snapshots.assert_called_once_with({})
        self.assertIn("sensor unavailable", robot._station_gripper_error)

    def test_publisher_cleanup_even_if_coordinator_stop_raises(self):
        robot = self.robot()
        robot._start_station_gripper_publisher()
        thread = robot._station_gripper_thread
        robot._coordinator = Mock()
        robot._coordinator.stop.side_effect = RuntimeError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            robot._stop_station_layer()
        self.assertFalse(thread.is_alive())
        self.assertIsNone(robot._station)

    def test_guard_disabled_does_not_start_snapshot_thread(self):
        robot = self.robot()
        robot._start_station_layer()
        self.assertIsNone(robot._station_gripper_thread)
        for gripper in robot._gripper.values():
            gripper.get_gripper_snapshot.assert_not_called()

    def test_geometry_backend_mismatch_is_rejected(self):
        robot = self.robot()
        station = SimpleNamespace(model=SimpleNamespace(names=["left_tool_taccap_right_finger_joint"]))
        robot._gripper["left"].config.type = "serial"
        with self.assertRaisesRegex(ValueError, "does not match"):
            robot._configure_station_grippers(station)

    def test_missing_calibration_is_rejected_before_any_hardware_connect(self):
        robot = self.robot()
        robot.config.native_collision_profile = "auto"
        robot._assembly = Mock()
        robot._configure_station_grippers = Mock(side_effect=ValueError("missing calibration"))
        robot._connect_arm = Mock()
        with (
            patch.dict(
                sys.modules,
                {
                    "libpyelite": SimpleNamespace(),
                    "libpyelite.collision": SimpleNamespace(StationLayer=Mock(return_value=Mock())),
                },
            ),
            patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt._import_elite_sdk") as sdk,
            self.assertRaisesRegex(ValueError, "missing calibration"),
        ):
            robot.connect()
        sdk.assert_not_called()
        robot._connect_arm.assert_not_called()
        for gripper in robot._gripper.values():
            gripper.connect.assert_not_called()

    def test_failed_connection_cleanup_joins_publisher_before_disconnecting_grippers(self):
        robot = self.robot()
        robot.logger = Mock()
        robot.cameras = {}
        robot._stop_servo_loop = Mock()
        robot._driver = dict.fromkeys(("left", "right"))
        robot._rtsi = dict.fromkeys(("left", "right"))
        robot._dashboard = dict.fromkeys(("left", "right"))
        robot._start_station_gripper_publisher()
        thread = robot._station_gripper_thread
        for gripper in robot._gripper.values():
            gripper._is_connected = True
            gripper.disconnect.side_effect = lambda: self.assertFalse(thread.is_alive())
        robot._cleanup_after_failed_connect()
        self.assertFalse(thread.is_alive())
        self.assertIsNone(robot._station)
        for gripper in robot._gripper.values():
            gripper.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
