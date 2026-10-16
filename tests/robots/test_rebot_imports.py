"""Other robots must remain importable even with an unusable B601 native SDK."""

import subprocess
import sys
import unittest


class TestRebotImports(unittest.TestCase):
    def test_robot_registration_does_not_load_native_b601_sdk(self):
        code = """
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path
from tempfile import TemporaryDirectory

class BrokenNativeSDK(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rebot_rt':
            raise OSError('B601 native library must not load for other robots')

sys.meta_path.insert(0, BrokenNativeSDK())
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.mock_robot import MockRobot, MockRobotConfig
with TemporaryDirectory() as directory:
    config = MockRobotConfig(calibration_dir=Path(directory))
    assert isinstance(make_robot_from_config(config), MockRobot)
assert 'rebot_b601_rt' in RobotConfig.get_known_choices()
assert 'bi_rebot_b601_rt' in RobotConfig.get_known_choices()
assert 'rebot_rt' not in sys.modules
assert 'lerobot.robots.rebot_b601_rt.rebot_b601_rt' not in sys.modules
assert 'lerobot.robots.bi_rebot_b601_rt.bi_rebot_b601_rt' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
