"""Arm-independent measured snapshots; SDKs and hardware are replaced locally."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lerobot.grippers.gripper import Gripper
from lerobot.grippers.serial.configuration_serial import SerialGripperConfig
from lerobot.grippers.taccap import taccap_follower as taccap_module
from lerobot.grippers.taccap.configuration_taccap import TaccapFollowerConfig
from lerobot.grippers.taccap.taccap_follower import TaccapFollower


class SnapshotTests(unittest.TestCase):
    def test_existing_backends_need_not_implement_snapshots(self):
        class LegacyGripper(Gripper):
            is_connected = False
            connect = Mock()
            disconnect = Mock()
            get_gripper_position = Mock(return_value=0.4)
            set_gripper_position = Mock()

        gripper = LegacyGripper(SimpleNamespace())
        self.assertFalse(gripper.get_gripper_snapshot().valid)
        self.assertEqual(gripper.get_gripper_position(), 0.4)
        gripper.connect.assert_not_called()

    def follower(self, raw=-0.6, age=20.0, seq=3, valid=True):
        gripper = TaccapFollower(TaccapFollowerConfig(side="left", init_open=False))
        gripper._is_connected = True
        gripper._snapshot_calibration = (0.1, 1.1, -1.0)
        observation = SimpleNamespace(raw_pos=raw, age_ms=age, seq=seq, valid=valid)
        gripper._loop = Mock()
        gripper._loop.snapshot.return_value = SimpleNamespace(running=True, observation=observation)
        self.addCleanup(setattr, gripper, "_is_connected", False)
        return gripper

    def test_taccap_actual_unclamped_reverse_calibration_and_age(self):
        gripper = self.follower()
        with patch("lerobot.grippers.taccap.taccap_follower.time.monotonic", return_value=100):
            sample = gripper.get_gripper_snapshot()
        self.assertTrue(sample.valid)
        self.assertAlmostEqual(sample.position, 0.5)
        self.assertAlmostEqual(sample.sampled_at, 99.98)
        self.assertEqual(sample.sequence, 3)
        self.assertEqual(sample.raw_position, -0.6)
        gripper._loop.snapshot.return_value.observation.raw_pos = -1.3
        self.assertAlmostEqual(gripper.get_gripper_snapshot().position, 1.2)
        gripper._loop.snapshot.return_value.observation.raw_pos = 0.1
        self.assertAlmostEqual(gripper.get_gripper_snapshot().position, -0.2)

    def test_taccap_sensor_is_read_without_observation_getter_or_device_io(self):
        gripper = self.follower()
        gripper.get_gripper_position = Mock(side_effect=AssertionError("legacy getter cache"))
        gripper._gripper = Mock()
        gripper._gripper.motor.read_status.side_effect = AssertionError("hardware IO")
        first = gripper.get_gripper_snapshot()
        gripper._loop.snapshot.return_value.observation.raw_pos = -0.9
        gripper._loop.snapshot.return_value.observation.seq = 4
        second = gripper.get_gripper_snapshot()
        self.assertNotEqual(first.position, second.position)
        self.assertEqual(second.sequence, 4)
        gripper._gripper.motor.read_status.assert_not_called()

    def test_taccap_invalid_missing_calibration_and_errors_never_use_cached_opening(self):
        for attributes in ({"valid": False}, {"seq": 0}, {"age_ms": -1}, {"age_ms": math.nan}, {"raw_pos": math.inf}):
            with self.subTest(attributes=attributes):
                gripper = self.follower()
                for key, value in attributes.items():
                    setattr(gripper._loop.snapshot.return_value.observation, key, value)
                self.assertFalse(gripper.get_gripper_snapshot().valid)
        gripper = self.follower()
        gripper._cached_position = 0.75
        gripper._loop.snapshot.side_effect = RuntimeError("read failed")
        self.assertFalse(gripper.get_gripper_snapshot().valid)
        gripper._snapshot_calibration = None
        self.assertFalse(gripper.get_gripper_snapshot().valid)

    def test_taccap_rollback_and_disconnect_clear_calibration(self):
        for method in ("_release_after_failed_connect", "disconnect"):
            gripper = self.follower()
            getattr(gripper, method)()
            self.assertIsNone(gripper._snapshot_calibration)
            self.assertFalse(gripper.get_gripper_snapshot().valid)

    def test_reconnect_with_failed_config_read_cannot_reuse_old_calibration(self):
        gripper = TaccapFollower(
            TaccapFollowerConfig(side="left", init_open=False, require_calibrated=False, mcu_device="/unused")
        )
        gripper._snapshot_calibration = (0, 1, 1)
        hardware = Mock()
        hardware.get_gripper_config.side_effect = RuntimeError("no config reply")
        loop = Mock()
        loop.snapshot.return_value = SimpleNamespace(
            grasp_torque_nm=1.0, hold_torque_limit_nm=2.0, observation=SimpleNamespace(position=0.4)
        )
        with (
            patch.object(taccap_module, "taccap", SimpleNamespace(FollowerGripper=lambda device: hardware)),
            patch.object(gripper, "_make_sdk_controller", return_value=loop),
        ):
            gripper.connect()
        self.addCleanup(setattr, gripper, "_is_connected", False)
        self.assertIsNone(gripper._snapshot_calibration)
        self.assertFalse(gripper.get_gripper_snapshot().valid)

    def test_optional_snapshot_metadata_does_not_change_legacy_connect_or_commands(self):
        for cfg in (
            SimpleNamespace(flags=taccap_module._CALIBRATED_FLAG),
            SimpleNamespace(flags=taccap_module._CALIBRATED_FLAG, min_open_rad=None, max_open_rad=1.0),
            SimpleNamespace(flags=taccap_module._CALIBRATED_FLAG, min_open_rad=0.0, max_open_rad=math.nan),
        ):
            with self.subTest(config=cfg):
                gripper = TaccapFollower(TaccapFollowerConfig(side="left", init_open=False, mcu_device="/unused"))
                hardware = Mock()
                hardware.get_gripper_config.return_value = cfg
                loop = Mock()
                loop.snapshot.return_value = SimpleNamespace(
                    grasp_torque_nm=1.0,
                    hold_torque_limit_nm=2.0,
                    observation=SimpleNamespace(position=0.4),
                )
                with (
                    patch.object(taccap_module, "taccap", SimpleNamespace(FollowerGripper=Mock(return_value=hardware))),
                    patch.object(gripper, "_make_sdk_controller", return_value=loop),
                ):
                    gripper.connect()
                self.addCleanup(setattr, gripper, "_is_connected", False)
                self.assertTrue(gripper.is_connected)
                hardware.motor.enable.assert_called_once()
                self.assertFalse(gripper.get_gripper_snapshot().valid)
                self.assertEqual(gripper.get_gripper_position(), 0.4)
                gripper.set_gripper_position(0.6)
                loop.set_target.assert_called_once_with(0.6)

    def serial_gripper(self):
        # Load a private driver module with only its SDK/discovery dependencies
        # replaced. Do not install a fake SDK into the rest of the test suite.
        path = Path(__file__).resolve().parents[2] / "src/lerobot/grippers/serial/serial_gripper.py"
        spec = importlib.util.spec_from_file_location("lerobot.grippers.serial._snapshot_test_driver", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "xgripper": SimpleNamespace(XenseSerialGripper=Mock()),
                "lerobot.grippers.serial.discovery": SimpleNamespace(find_port_by_side=Mock(), find_port_by_sn=Mock()),
            },
        ):
            spec.loader.exec_module(module)
        gripper = module.SerialGripper(SerialGripperConfig(port="/unused", init_open=False))
        self.addCleanup(setattr, gripper, "_is_connected", False)
        return gripper

    @staticmethod
    def poll_once(gripper, raw_position):
        gripper._poll_running = True

        def reply(**kwargs):
            gripper._poll_running = False
            return {"position": raw_position}

        gripper._gripper = Mock()
        gripper._gripper.get_gripper_status.side_effect = reply
        gripper._gripper.get_receive_thread_failure_reason.return_value = None
        gripper._position_poll_loop()

    def test_serial_cache_is_invalid_until_real_reply_and_timestamp_does_not_refresh(self):
        gripper = self.serial_gripper()
        self.assertFalse(gripper.get_gripper_snapshot().valid)
        gripper._is_connected = True
        self.poll_once(gripper, 42.5)
        first = gripper.get_gripper_snapshot()
        self.assertTrue(first.valid)
        self.assertAlmostEqual(first.position, 0.5)
        self.assertGreater(first.sampled_at, 0)
        self.assertIs(first, gripper.get_gripper_snapshot())
        gripper._cached_position = 0.9  # a legacy getter/command cache is not sensor data
        self.assertIs(first, gripper.get_gripper_snapshot())

    def test_serial_snapshot_preserves_overtravel_while_legacy_position_stays_clamped(self):
        gripper = self.serial_gripper()
        gripper._is_connected = True
        for raw, opening, legacy in ((-8.5, -0.1, 0.0), (93.5, 1.1, 1.0)):
            with self.subTest(raw=raw):
                self.poll_once(gripper, raw)
                sample = gripper.get_gripper_snapshot()
                self.assertTrue(sample.valid)
                self.assertAlmostEqual(sample.position, opening)
                self.assertEqual(sample.raw_position, raw)
                self.assertEqual(gripper.get_gripper_position(), legacy)

    def test_serial_missing_reply_keeps_sample_age_and_disconnect_invalidates_it(self):
        gripper = self.serial_gripper()
        gripper._is_connected = True
        self.poll_once(gripper, 42.5)
        first = gripper.get_gripper_snapshot()
        self.poll_once(gripper, None)
        self.assertIs(gripper.get_gripper_snapshot(), first)
        gripper._is_connected = False
        self.assertFalse(gripper.get_gripper_snapshot().valid)


if __name__ == "__main__":
    unittest.main()
