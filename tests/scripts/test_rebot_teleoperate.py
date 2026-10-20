"""B601 launch recipes seed Cartesian teleoperators from the measured tool pose."""

import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import draccus
import numpy as np

from lerobot.scripts import lerobot_teleoperate as cli, teleop_device_loops as loops
from tests.scripts.conftest_teleop_doubles import FakeRobot, FakeTeleop, run_loop

ROOT = Path(__file__).resolve().parents[2]


class TestRebotTeleoperate(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(cli, "logger"))

    def test_single_arm_recipe_and_pico4_startup(self):
        config = draccus.parse(
            cli.TeleoperateConfig,
            config_path=ROOT / "recipes/teleop/rebot_b601_rt/rs-taccap.yaml",
            args=["--display_data=false", "--dryrun=true"],
        )
        self.assertEqual(config.robot.motor_family, "rs")
        self.assertEqual(config.robot.control_mode, "pos_vel")
        self.assertTrue(config.robot.interpolate_cartesian_targets)
        self.assertFalse(config.robot.mit_velocity_feedforward)
        self.assertTrue(config.robot.reset_on_exit)
        self.assertEqual(config.robot.reset_position_rad, [0] * 6)
        self.assertEqual(config.robot.max_joint_speed, 1.5)
        self.assertEqual(config.robot.reset_joint_speed, 0.5)
        self.assertTrue(config.robot.uses_taccap_model)
        self.assertFalse(config.robot.has_gripper)
        pose = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 0.7])
        robot = Mock(get_current_tcp_pose_quat=Mock(return_value=pose))
        teleop = Mock()
        with (
            patch.object(cli, "make_robot_from_config", return_value=robot),
            patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
            patch.object(cli, "run_cartesian_teleop_loop") as loop,
            patch.object(cli, "_cleanup"),
        ):
            cli.teleoperate(config)
        robot.connect.assert_called_once_with()
        np.testing.assert_allclose(teleop.connect.call_args.kwargs["current_tcp_pose_quat"], pose)
        self.assertIsInstance(loop.call_args.kwargs["policy"], cli.Pico4Policy)
        self.assertTrue(loop.call_args.kwargs["dryrun"])
        robot.disconnect.assert_called_once_with(reset=False)

    def test_bimanual_recipe_waits_for_vr_then_seeds_each_arm(self):
        config = draccus.parse(
            cli.TeleoperateConfig,
            config_path=ROOT / "recipes/teleop/bi_rebot_b601_rt/rs-taccap.yaml",
            args=["--display_data=false"],
        )
        self.assertEqual(config.robot.left_arm_config.channel, "can0")
        self.assertEqual(config.robot.right_arm_config.channel, "can1")
        self.assertTrue(config.robot.reset_on_exit)
        for arm in (config.robot.left_arm_config, config.robot.right_arm_config):
            self.assertEqual(arm.motor_family, "rs")
            self.assertEqual(arm.control_mode, "pos_vel")
            self.assertTrue(arm.interpolate_cartesian_targets)
            self.assertFalse(arm.mit_velocity_feedforward)
            self.assertEqual(arm.reset_position_rad, [0] * 6)
            self.assertEqual(arm.max_joint_speed, 1.5)
            self.assertEqual(arm.reset_joint_speed, 0.5)
            self.assertTrue(arm.uses_taccap_model)
            self.assertFalse(arm.has_gripper)
        left = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 0.7])
        right = np.array([0.4, 0.5, 0.6, 1, 0, 0, 0, 0.3])
        events = []
        robot = Mock(get_current_tcp_pose_quat=Mock(return_value=(left, right)))
        robot.connect.side_effect = lambda: events.append("arm")
        teleop = Mock()
        teleop.pre_init.side_effect = lambda: events.append("vr")
        with (
            patch.object(cli, "make_robot_from_config", return_value=robot),
            patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
            patch.object(cli, "run_cartesian_teleop_loop") as loop,
            patch.object(cli, "_cleanup"),
        ):
            cli.teleoperate(config)
        self.assertEqual(events, ["vr", "arm"])
        np.testing.assert_allclose(teleop.connect.call_args.kwargs["left_tcp_pose_quat"], left)
        np.testing.assert_allclose(teleop.connect.call_args.kwargs["right_tcp_pose_quat"], right)
        self.assertIsInstance(loop.call_args.kwargs["policy"], cli.BiPico4Policy)
        robot.disconnect.assert_called_once_with(reset=True)

    def test_disconnect_requests_return_only_after_normal_or_interrupted_loop(self):
        for mode in ("mit", "pos_vel"):
            for outcome, enabled, dryrun, expected in (
                (None, True, False, True),
                (KeyboardInterrupt(), True, False, True),
                (RuntimeError("motor fault"), True, False, False),
                (KeyboardInterrupt(), False, False, True),
                (KeyboardInterrupt(), True, True, False),
            ):
                with self.subTest(mode=mode, outcome=outcome, enabled=enabled, dryrun=dryrun):
                    config = draccus.parse(
                        cli.TeleoperateConfig,
                        config_path=ROOT / "recipes/teleop/rebot_b601_rt/rs-taccap.yaml",
                        args=["--display_data=false", f"--robot.control_mode={mode}"],
                    )
                    config.robot.reset_on_exit = enabled
                    config.dryrun = dryrun
                    robot, teleop = Mock(), Mock()
                    events = Mock()
                    events.attach_mock(robot.disconnect, "disconnect")
                    with (
                        patch.object(cli, "make_robot_from_config", return_value=robot),
                        patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
                        patch.object(cli, "run_cartesian_teleop_loop", side_effect=outcome),
                        patch.object(cli, "_cleanup", new=events.cleanup),
                    ):
                        cli.teleoperate(config)
                    self.assertEqual(
                        events.mock_calls, [call.disconnect(reset=expected), call.cleanup(robot, teleop, False)]
                    )

    def test_startup_interrupt_does_not_start_exit_reset(self):
        config = draccus.parse(
            cli.TeleoperateConfig,
            config_path=ROOT / "recipes/teleop/rebot_b601_rt/rs-taccap.yaml",
            args=["--display_data=false"],
        )
        robot, teleop = Mock(), Mock()
        teleop.connect.side_effect = KeyboardInterrupt
        with (
            patch.object(cli, "make_robot_from_config", return_value=robot),
            patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
            patch.object(cli, "_cleanup") as cleanup,
            self.assertRaises(KeyboardInterrupt),
        ):
            cli.teleoperate(config)
        robot.disconnect.assert_called_once_with(reset=False)
        cleanup.assert_called_once_with(robot, teleop, False)

    def test_single_arm_spacemouse_uses_measured_euler_pose(self):
        config = draccus.parse(
            cli.TeleoperateConfig,
            config_path=ROOT / "recipes/teleop/rebot_b601_rt/rs-taccap.yaml",
            args=["--teleop.type=spacemouse", "--display_data=false"],
        )
        pose = np.array([0.1, 0.2, 0.3, 0, 0, 1.5, 0.7])
        robot = Mock(get_current_tcp_pose_euler=Mock(return_value=pose))
        teleop = Mock()
        with (
            patch.object(cli, "make_robot_from_config", return_value=robot),
            patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
            patch.object(cli, "run_cartesian_teleop_loop") as loop,
            patch.object(cli, "_cleanup"),
        ):
            cli.teleoperate(config)
        np.testing.assert_allclose(teleop.connect.call_args.kwargs["current_tcp_pose_euler"], pose)
        self.assertIsInstance(loop.call_args.kwargs["policy"], cli.SpaceMousePolicy)

    def test_a_reset_stands_aside_then_resyncs_before_sending_again(self):
        robot = FakeRobot(iterations=5, name="rebot_b601_rt", rt_moving_frames={1, 2})
        teleop = FakeTeleop(reset_frames={0})
        with (
            patch.object(loops, "precise_sleep"),
            patch.object(loops, "_sleep_to_fps"),
            patch.object(cli, "_print_obs_state"),
            patch("builtins.print"),
        ):
            run_loop(
                loops.run_cartesian_teleop_loop,
                iterations=5,
                teleop=teleop,
                robot=robot,
                fps=30,
                policy=cli.Pico4Policy(),
            )
        self.assertEqual(robot.names().count("reset_to_initial_position"), 1)
        self.assertEqual(robot.names().count("send_action"), 1)
        self.assertEqual(teleop.names().count("reset_to_pose"), 1)
        self.assertLess(robot.names().index("get_current_tcp_pose_quat"), robot.names().index("send_action"))


if __name__ == "__main__":
    unittest.main()
