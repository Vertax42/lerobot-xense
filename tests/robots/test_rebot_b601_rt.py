"""Offline model selection and independent gripper routing for B601 RT arms."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import draccus
import numpy as np

try:
    import rebot_rt
except ImportError as exc:
    raise unittest.SkipTest("Build libpyrebot to run the B601 RT adapter tests") from exc

from lerobot.grippers import SerialGripperConfig, TaccapFollowerConfig
from lerobot.robots.bi_rebot_b601_rt import BiRebotB601RT, BiRebotB601RTConfig
from lerobot.robots.rebot_b601_rt import RebotB601RT, RebotB601RTConfig, rebot_b601_rt as driver

ASSETS = Path(rebot_rt.__file__).parent / "assets"
TACCAP_URDF = "RS_Taccap/urdf/rebot_b601_rs_taccap_v2_gripper.urdf"


def fake_gripper():
    gripper = SimpleNamespace(is_connected=False)
    gripper.connect = Mock(side_effect=lambda: setattr(gripper, "is_connected", True))
    gripper.disconnect = Mock(side_effect=lambda: setattr(gripper, "is_connected", False))
    gripper.get_gripper_position = Mock(return_value=0.73)
    gripper.set_gripper_position = Mock()
    return gripper


def fake_native():
    native = SimpleNamespace(connected=False)
    native.connect = Mock(side_effect=lambda: setattr(native, "connected", True))
    native.start = Mock()
    native.stop = Mock()
    native.close = Mock(side_effect=lambda: setattr(native, "connected", False))
    native.get_state = Mock(
        return_value=SimpleNamespace(
            fault=False,
            q=[0.0] * 6 + [-1.2],
            command=[0.0] * 6 + [-1.2],
            target=[0.0] * 6 + [-1.2],
            command_velocity=[0.0] * 7,
            mit_command_velocity=[0.0] * 7,
            cartesian_ik_valid=False,
        )
    )
    native.get_target_joints = Mock(return_value=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.2])
    native.set_target_joints = Mock()
    native.queue_target_pose = Mock(return_value=1)
    native.hold_position = Mock()
    native.forward_kinematics = Mock(return_value=np.eye(4))
    native.set_target_pose = Mock(return_value=SimpleNamespace(success=True, target=[0.0] * 6 + [-1.2]))
    return native


class TestRebotB601RT(unittest.TestCase):
    def setUp(self):
        temp_dir = self.enterContext(TemporaryDirectory())
        self.calibration_dir = Path(temp_dir)
        self.native = fake_native()
        self.native_factory = self.enterContext(patch.object(driver.rebot_rt, "Robot", return_value=self.native))
        self.gripper_factory = self.enterContext(patch.object(driver, "make_gripper_from_config", return_value=None))

    def make_robot(self, **kwargs):
        kwargs.setdefault("interpolate_cartesian_targets", False)
        config = RebotB601RTConfig(calibration_dir=self.calibration_dir, **kwargs)
        robot = RebotB601RT(config)
        self.addCleanup(lambda: robot.disconnect() if robot.is_connected else None)
        return robot

    def test_native_cartesian_queue_in_both_control_modes(self):
        self.assertTrue(RebotB601RTConfig().interpolate_cartesian_targets)
        for mode in ("mit", "pos_vel"):
            with self.subTest(mode=mode):
                robot = self.make_robot(control_mode=mode, has_gripper=False, interpolate_cartesian_targets=True)
                robot.connect()
                robot.logger = Mock()
                action = driver._action_from_pose(np.eye(4))
                robot.send_action(action)
                self.native.queue_target_pose.assert_called()
                self.native.set_target_pose.assert_not_called()
                state = self.native.get_state.return_value
                state.cartesian_ik_valid = True
                state.cartesian_ik_success = False
                state.cartesian_ik_error = 0.1
                robot.get_observation()
                robot.logger.warn.assert_called_once()
                state.cartesian_ik_success = True
                robot.get_observation()
                robot.logger.info.assert_called_once()
                robot.disconnect()

    def test_partial_tracking_is_not_reported_as_an_ik_rejection(self):
        robot = self.make_robot(has_gripper=False, interpolate_cartesian_targets=True)
        robot.connect()
        robot.logger = Mock()
        state = self.native.get_state.return_value
        state.cartesian_ik_valid = True
        state.cartesian_ik_success = False
        state.cartesian_tracking_limited = True
        state.cartesian_tracking_progress = True
        with patch.object(driver.time, "monotonic", return_value=10.0):
            robot.get_observation()
        robot.logger.warn.assert_not_called()
        self.assertIn("approaching", robot.logger.info.call_args.args[0])
        # Limit transitions can occur at every producer update. Do not flood
        # the terminal when tracking alternates between partial and complete.
        with patch.object(driver.time, "monotonic", return_value=10.1):
            state.cartesian_tracking_limited = False
            robot.get_observation()
            state.cartesian_tracking_limited = True
            robot.get_observation()
        self.assertEqual(robot.logger.info.call_count, 1)
        with patch.object(driver.time, "monotonic", return_value=11.1):
            state.cartesian_tracking_limited = False
            robot.get_observation()
        self.assertIn("reached", robot.logger.info.call_args.args[0])

    def test_motor_family_and_gripper_select_models_independently(self):
        for family in ("dm", "RS"):
            for taccap in (False, True):
                with self.subTest(family=family, taccap=taccap):
                    gripper = fake_gripper() if taccap else None
                    self.gripper_factory.return_value = gripper
                    robot = self.make_robot(
                        motor_family=family,
                        gripper=TaccapFollowerConfig(auto_discover_cameras=False) if taccap else None,
                    )
                    robot.connect()
                    args = self.native_factory.call_args.args
                    model = TACCAP_URDF if taccap else f"{family.upper()}/urdf/ReBot_Arm_{family.upper()}.urdf"
                    self.assertEqual(args[0], rebot_rt.Family.DM if family == "dm" else rebot_rt.Family.RS)
                    self.assertEqual(Path(args[2]), ASSETS / model)
                    self.assertTrue(Path(args[2]).is_file())
                    self.assertEqual(args[3], "taccap_eef_link" if taccap else "")
                    self.assertEqual(args[4], not taccap)
                    self.assertEqual(args[9], "pos_vel")
                    self.assertIn("gripper.pos", robot.action_features)
                    self.assertIn("gripper.pos", robot.observation_features)
                    obs = robot.get_observation()
                    self.assertEqual(obs["gripper.pos"], 0.73 if taccap else -1.2)
                    action = {**driver._action_from_pose(np.eye(4)), "gripper.pos": 0.4 if taccap else -0.4}
                    sent = robot.send_action(action)
                    self.assertEqual(self.native.set_target_pose.call_args.args[1], 0.0 if taccap else -0.4)
                    if taccap:
                        gripper.set_gripper_position.assert_called_once_with(0.4)
                        self.assertEqual(sent["gripper.pos"], 0.4)
                    robot.disconnect()
                    if taccap:
                        gripper.connect.assert_called_once()
                        gripper.disconnect.assert_called_once()

    def test_recipe_decodes_a_taccap_block_for_both_families(self):
        for family in ("DM", "rs"):
            with self.subTest(family=family):
                config = draccus.decode(
                    RebotB601RTConfig,
                    {
                        "motor_family": family,
                        "gripper": {"type": "taccap_follower", "side": "right"},
                    },
                )
                self.assertEqual(config.default_urdf, TACCAP_URDF)
                self.assertEqual(config.gripper.side, "right")
                self.assertFalse(config.has_gripper)

    def test_control_mode_and_position_gains_can_be_overridden(self):
        robot = self.make_robot(
            control_mode="MIT", mit_velocity_feedforward=False, motor_configs=[{"vel_kp": 7.0}, {}, {}, {}, {}, {}, {}]
        )
        robot.connect()
        args = self.native_factory.call_args.args
        self.assertEqual(args[9], "mit")
        self.assertFalse(args[10])
        self.assertEqual(args[8][0].vel_kp, 7.0)
        with self.assertRaisesRegex(ValueError, "control_mode"):
            RebotB601RTConfig(control_mode="invalid")

    def test_six_motor_arm_and_serial_gripper_keep_original_model(self):
        for family in ("dm", "rs"):
            for gripper_config in (None, SerialGripperConfig(side="left", auto_discover_cameras=False)):
                with self.subTest(family=family, gripper=gripper_config):
                    self.gripper_factory.return_value = fake_gripper() if gripper_config else None
                    robot = self.make_robot(motor_family=family, has_gripper=False, gripper=gripper_config)
                    robot.connect()
                    self.assertEqual(
                        Path(self.native_factory.call_args.args[2]),
                        ASSETS / family.upper() / "urdf" / f"ReBot_Arm_{family.upper()}.urdf",
                    )
                    self.assertEqual("gripper.pos" in robot.action_features, gripper_config is not None)
                    robot.disconnect()

    def test_explicit_model_and_tool_frame_override_taccap_defaults(self):
        self.gripper_factory.return_value = fake_gripper()
        urdf = ASSETS / "DM/urdf/ReBot_Arm_DM.urdf"
        robot = self.make_robot(
            motor_family="rs",
            urdf_path=str(urdf),
            end_frame="link6",
            gripper=TaccapFollowerConfig(auto_discover_cameras=False),
        )
        robot.connect()
        self.assertEqual(self.native_factory.call_args.args[2:5], (str(urdf), "link6", False))

    def test_explicit_taccap_model_disables_native_gripper_for_both_families(self):
        for family in ("dm", "rs"):
            robot = self.make_robot(motor_family=family, urdf_path=str(ASSETS / TACCAP_URDF))
            robot.connect()
            self.assertEqual(self.native_factory.call_args.args[3:5], ("taccap_eef_link", False))
            self.assertNotIn("gripper.pos", robot.action_features)
            robot.disconnect()

    def test_external_gripper_cannot_also_enable_seventh_arm_motor(self):
        with self.assertRaisesRegex(ValueError, "has_gripper=False"):
            RebotB601RTConfig(gripper=TaccapFollowerConfig(), has_gripper=True)

    def test_bad_external_target_does_not_command_the_arm(self):
        self.gripper_factory.return_value = fake_gripper()
        robot = self.make_robot(gripper=TaccapFollowerConfig(auto_discover_cameras=False))
        robot.connect()
        for target in (-0.1, 1.1, float("nan")):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                robot.send_action({**driver._action_from_pose(np.eye(4)), "gripper.pos": target})
        self.native.set_target_pose.assert_not_called()

    def test_ik_rejection_holds_last_target_and_recovers_without_disconnecting(self):
        gripper = fake_gripper()
        self.gripper_factory.return_value = gripper
        robot = self.make_robot(gripper=TaccapFollowerConfig(auto_discover_cameras=False))
        robot.connect()
        robot.logger = Mock()
        held_pose = np.eye(4)
        held_pose[:3, 3] = [0.1, 0.2, 0.3]
        self.native.forward_kinematics.return_value = held_pose
        self.native.set_target_pose.return_value = SimpleNamespace(
            success=False,
            error=0.00101552,
            target=[99.0] * 7,
        )
        action = {**driver._action_from_pose(np.eye(4)), "gripper.pos": 0.4}
        with patch.object(driver.time, "monotonic", return_value=10.0):
            for _ in range(2):
                sent = robot.send_action(action)
        self.assertTrue(robot.is_connected)
        self.assertTrue(robot._ik_failed)
        self.native.forward_kinematics.assert_called_with(self.native.get_target_joints.return_value)
        np.testing.assert_allclose([sent[k] for k in ("tcp.x", "tcp.y", "tcp.z")], [0.1, 0.2, 0.3])
        self.assertEqual(sent["gripper.pos"], 0.4)
        gripper.set_gripper_position.assert_called_with(0.4)
        robot.logger.warn.assert_called_once()
        self.native.set_target_pose.return_value = SimpleNamespace(success=True, target=[0.0] * 7)
        robot.send_action(action)
        self.assertFalse(robot._ik_failed)
        robot.logger.info.assert_called_once()

    def test_ik_rejection_does_not_hide_a_motor_fault(self):
        robot = self.make_robot(has_gripper=False)
        robot.connect()
        self.native.set_target_pose.return_value = SimpleNamespace(success=False, error=0.001)
        self.native.get_state.return_value = SimpleNamespace(fault=True, fault_reason="feedback lost")
        with self.assertRaisesRegex(RuntimeError, "feedback lost"):
            robot.send_action(driver._action_from_pose(np.eye(4)))
        self.native.get_target_joints.assert_not_called()

    def test_reset_returns_to_configured_joints_and_waits_for_measured_settling(self):
        robot = self.make_robot(reset_position_rad=[0.0] * 6, max_joint_speed=1.5, reset_joint_speed=0.5)
        robot.connect()
        with patch.object(driver.time, "monotonic", return_value=0.0) as clock:
            robot.reset_to_initial_position()
            self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.5)
            self.native.get_state.return_value = SimpleNamespace(
                fault=False,
                q=[0.2] * 6 + [-1.2],
                command=[0.1] * 6 + [-1.2],
            )
            clock.return_value = 0.2
            self.assertTrue(robot.rt_moving)
            with self.assertRaisesRegex(RuntimeError, "reset is in progress"):
                robot.send_action(driver._action_from_pose(np.eye(4)))
            self.native.get_state.return_value.command = [0.0] * 6 + [-1.2]
            self.assertTrue(robot.rt_moving)  # Command arrived; measured arm has not.
            self.native.get_state.return_value.q = [0.001] * 6 + [-1.2]
            clock.return_value = 0.3
            self.assertTrue(robot.rt_moving)
            clock.return_value = 0.41
            self.assertFalse(robot.rt_moving)
            self.assertIsNone(robot._reset_target)
        self.native.set_target_pose.assert_not_called()
        # Every watchdog refresh must retain the slower reset speed.
        for request in self.native.set_target_joints.call_args_list:
            self.assertEqual(request.kwargs, {"max_joint_speed": 0.5})

    def test_reset_speed_never_exceeds_the_arm_limit(self):
        robot = self.make_robot(reset_position_rad=[0.0] * 6, max_joint_speed=0.25)
        robot.connect()
        robot.reset_to_initial_position()
        self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.25)
        self.assertTrue(robot.rt_moving)
        self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.25)

    def test_joint_speed_configuration_requires_finite_positive_values(self):
        for field in ("max_joint_speed", "reset_joint_speed"):
            for speed in (0, -0.1, float("nan"), float("inf")):
                with self.subTest(field=field, speed=speed), self.assertRaisesRegex(ValueError, "positive"):
                    RebotB601RTConfig(**{field: speed})
        for limits in ([], [1.0] * 5, [1.0] * 7, [0.0] * 6, [-1.0] * 6, [float("nan")] * 6):
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "max_joint_speeds"):
                RebotB601RTConfig(max_joint_speeds=limits)

    def test_joint_limits_and_cartesian_coordination_reach_native_in_both_modes(self):
        limits = [0.2, 0.8, 1.1, 0.9, 0.7, 0.6]
        for mode in ("mit", "pos_vel"):
            with self.subTest(mode=mode):
                robot = self.make_robot(control_mode=mode, max_joint_speeds=limits)
                robot.connect()
                self.assertEqual(
                    self.native_factory.call_args.kwargs,
                    {
                        "max_joint_speeds": limits,
                        "coordinate_cartesian_speed": True,
                    },
                )
                robot.disconnect()

    def test_reset_caps_using_per_joint_limits_instead_of_unused_scalar(self):
        robot = self.make_robot(reset_position_rad=[0.0] * 6, max_joint_speed=1.5, max_joint_speeds=[0.25] * 6)
        robot.connect()
        robot.reset_to_initial_position()
        self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.25)
        self.assertTrue(robot.rt_moving)
        self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.25)

    def test_dual_arm_decoding_keeps_independent_speed_limits(self):
        left, right = [0.2, 0.8, 1.1, 0.9, 0.7, 0.6], [0.4] * 6
        config = draccus.decode(
            BiRebotB601RTConfig,
            {
                "left_arm_config": {"max_joint_speeds": left, "coordinate_cartesian_speed": True},
                "right_arm_config": {"channel": "can1", "max_joint_speeds": right, "coordinate_cartesian_speed": False},
            },
        )
        self.assertEqual(config.left_arm_config.max_joint_speeds, left)
        self.assertEqual(config.right_arm_config.max_joint_speeds, right)
        self.assertTrue(config.left_arm_config.coordinate_cartesian_speed)
        self.assertFalse(config.right_arm_config.coordinate_cartesian_speed)

    def test_reset_timeout_holds_current_command(self):
        robot = self.make_robot(reset_position_rad=[0.0] * 6, reset_timeout_sec=1.0)
        robot.connect()
        with patch.object(driver.time, "monotonic", return_value=0.0) as clock:
            robot.reset_to_initial_position()
            clock.return_value = 1.1
            with self.assertRaisesRegex(TimeoutError, "reset_timeout_sec"):
                _ = robot.rt_moving
        self.native.hold_position.assert_called_once()
        self.assertIsNone(robot._reset_target)

    def test_reset_needs_a_configured_six_joint_pose(self):
        robot = self.make_robot()
        robot.connect()
        with self.assertRaisesRegex(ValueError, "reset_position_rad"):
            robot.reset_to_initial_position()
        self.native.set_target_joints.assert_not_called()
        for angles in ([0.0] * 5, [float("nan")] * 6):
            with self.assertRaisesRegex(ValueError, "six finite"):
                RebotB601RTConfig(reset_position_rad=angles)

    def test_exit_reset_config_requires_explicit_poses(self):
        self.assertFalse(RebotB601RTConfig().reset_on_exit)
        with self.assertRaisesRegex(ValueError, "reset_on_exit"):
            RebotB601RTConfig(reset_on_exit=True)
        for missing_side in ("left", "right"):
            with self.subTest(side=missing_side), self.assertRaisesRegex(ValueError, "both arms"):
                BiRebotB601RTConfig(
                    reset_on_exit=True,
                    left_arm_config=RebotB601RTConfig(
                        channel="can0", reset_position_rad=None if missing_side == "left" else [0] * 6
                    ),
                    right_arm_config=RebotB601RTConfig(
                        channel="can1", reset_position_rad=None if missing_side == "right" else [0] * 6
                    ),
                )

    def test_reset_checks_native_fault_before_sending_goal(self):
        robot = self.make_robot(reset_position_rad=[0.0] * 6)
        robot.connect()
        self.native.get_state.return_value.fault = True
        self.native.get_state.return_value.fault_reason = "motor fault"
        with self.assertRaisesRegex(RuntimeError, "motor fault"):
            robot.reset_to_initial_position()
        self.native.set_target_joints.assert_not_called()
        # Also surface faults after this arm has settled, while a second arm
        # might still be returning in the shared exit-reset polling loop.
        with self.assertRaisesRegex(RuntimeError, "motor fault"):
            _ = robot.rt_moving

    def test_tracking_trace_separates_limiting_from_motor_tracking_error(self):
        robot = self.make_robot(trace_joint_tracking=True, control_mode="mit", mit_velocity_feedforward=False)
        robot.connect()
        robot.logger = Mock()
        self.native.get_state.return_value = SimpleNamespace(
            fault=False,
            q=[0.1] * 7,
            target=[0.4] * 7,
            command=[0.2] * 7,
            command_velocity=[0.5] * 7,
            mit_command_velocity=[0.0] * 7,
        )
        robot.get_observation()
        message = robot.logger.info.call_args.args[0]
        self.assertIn("goal=[0.4", message)
        self.assertIn("command=[0.2", message)
        self.assertIn("measured=[0.1", message)
        self.assertIn("velocity=[0.5", message)
        self.assertIn("mit_dq=[0.0", message)

    def test_teleop_pose_uses_wxyz_quaternion_and_external_gripper(self):
        self.gripper_factory.return_value = fake_gripper()
        pose = np.array([[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]])
        self.native.forward_kinematics.return_value = pose
        robot = self.make_robot(
            use_joint_observation=True,
            gripper=TaccapFollowerConfig(auto_discover_cameras=False),
        )
        robot.connect()
        np.testing.assert_allclose(
            robot.get_current_tcp_pose_quat(),
            [0.1, 0.2, 0.3, 2**-0.5, 0.0, 0.0, 2**-0.5, 0.73],
        )
        np.testing.assert_allclose(robot.get_current_tcp_pose_euler(), [0.1, 0.2, 0.3, 0, 0, np.pi / 2, 0.73])

    def test_failed_camera_connect_releases_arm_and_external_gripper(self):
        gripper = fake_gripper()
        self.gripper_factory.return_value = gripper
        camera = SimpleNamespace(is_connected=False, connect=Mock(side_effect=RuntimeError("camera failed")))
        self.enterContext(patch.object(driver, "make_cameras_from_configs", return_value={"scene": camera}))
        robot = self.make_robot(gripper=TaccapFollowerConfig(auto_discover_cameras=False))
        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            robot.connect()
        self.native.close.assert_called_once()
        gripper.disconnect.assert_called_once()
        self.assertFalse(robot.is_connected)

    def test_taccap_camera_discovery_pins_the_separate_gripper(self):
        gripper = fake_gripper()
        self.gripper_factory.return_value = gripper
        inject = self.enterContext(patch.object(driver, "inject_taccap_cameras", return_value={"right": "/dev/fake"}))
        self.make_robot(motor_family="DM", gripper=TaccapFollowerConfig(side="right"))
        self.assertEqual(inject.call_args.kwargs["sides"], ("right",))
        self.assertEqual(gripper._mcu_device, "/dev/fake")

    def test_bimanual_arms_choose_their_own_model_and_gripper(self):
        self.gripper_factory.side_effect = [fake_gripper(), None]
        self.native_factory.side_effect = [fake_native(), fake_native()]
        config = BiRebotB601RTConfig(
            calibration_dir=self.calibration_dir,
            left_arm_config=RebotB601RTConfig(
                motor_family="dm",
                channel="can0",
                gripper=TaccapFollowerConfig(side="left", auto_discover_cameras=False),
            ),
            right_arm_config=RebotB601RTConfig(motor_family="RS", channel="can1"),
        )
        robot = BiRebotB601RT(config)
        try:
            robot.connect()
            left, right = self.native_factory.call_args_list
            self.assertEqual(Path(left.args[2]), ASSETS / TACCAP_URDF)
            self.assertEqual(Path(right.args[2]), ASSETS / "RS/urdf/ReBot_Arm_RS.urdf")
            obs = robot.get_observation()
            self.assertEqual(obs["left_gripper.pos"], 0.73)
            self.assertEqual(obs["right_gripper.pos"], -1.2)
            left_pose, right_pose = robot.get_current_tcp_pose_quat()
            np.testing.assert_allclose(left_pose, [0, 0, 0, 1, 0, 0, 0, 0.73])
            np.testing.assert_allclose(right_pose, [0, 0, 0, 1, 0, 0, 0, -1.2])
        finally:
            if robot.is_connected:
                robot.disconnect()

    def test_bimanual_reset_refreshes_both_watchdogs_and_finishes_together(self):
        left, right = fake_native(), fake_native()
        self.native_factory.side_effect = [left, right]
        config = BiRebotB601RTConfig(
            calibration_dir=self.calibration_dir,
            left_arm_config=RebotB601RTConfig(channel="can0", reset_position_rad=[0.0] * 6),
            right_arm_config=RebotB601RTConfig(channel="can1", reset_position_rad=[0.0] * 6),
        )
        robot = BiRebotB601RT(config)
        try:
            robot.connect()
            with patch.object(driver.time, "monotonic", return_value=0.0) as clock:
                robot.reset_to_initial_position()
                self.assertTrue(robot.rt_moving)
                self.assertGreaterEqual(left.set_target_joints.call_count, 2)
                self.assertGreaterEqual(right.set_target_joints.call_count, 2)
                clock.return_value = 0.11
                self.assertFalse(robot.rt_moving)
        finally:
            if robot.is_connected:
                robot.disconnect()

    def test_bimanual_reset_checks_both_arms_before_starting_either(self):
        left, right = fake_native(), fake_native()
        shutdown = Mock()
        shutdown.attach_mock(left.stop, "left_stop")
        shutdown.attach_mock(right.stop, "right_stop")
        shutdown.attach_mock(left.close, "left_close")
        shutdown.attach_mock(right.close, "right_close")
        self.native_factory.side_effect = [left, right]
        robot = BiRebotB601RT(
            BiRebotB601RTConfig(
                calibration_dir=self.calibration_dir,
                left_arm_config=RebotB601RTConfig(channel="can0", reset_position_rad=[0] * 6),
                right_arm_config=RebotB601RTConfig(channel="can1", reset_position_rad=[0] * 6),
            )
        )
        robot.connect()
        try:
            right.get_state.return_value.fault = True
            right.get_state.return_value.fault_reason = "right arm fault"
            with self.assertRaisesRegex(RuntimeError, "right arm fault"):
                robot.reset_to_initial_position()
            left.set_target_joints.assert_not_called()
            right.set_target_joints.assert_not_called()
        finally:
            robot.disconnect()
        left.stop.assert_called_once()
        right.stop.assert_called_once()
        self.assertEqual(
            shutdown.mock_calls, [call.left_stop(), call.right_stop(), call.right_close(), call.left_close()]
        )


if __name__ == "__main__":
    unittest.main()
