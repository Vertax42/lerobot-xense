#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract gripper interface and implementations for Flexiv Rizon4 robot."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GripperType(str, Enum):
    """Supported gripper types."""

    NONE = "none"  # No gripper
    FLEXIV_GRAV = "flexiv_grav"  # Flexiv Grav gripper (via flexivrdk)
    # Add more gripper types here as needed


@dataclass
class GripperConfig:
    """Base configuration for grippers.

    Attributes:
        gripper_type: Type of gripper to use
        open_width: Fully open width in meters
        close_width: Fully closed width in meters
    """

    gripper_type: GripperType = GripperType.NONE
    open_width: float = 0.087  # m, fully open position
    close_width: float = 0.0  # m, fully closed position


@dataclass
class FlexivGravGripperConfig(GripperConfig):
    """Configuration for Flexiv Grav gripper.

    Attributes:
        speed: Gripper moving speed in m/s
        force: Gripping force in N
    """

    gripper_type: GripperType = field(default=GripperType.FLEXIV_GRAV, init=False)
    speed: float = 0.1  # m/s
    force: float = 20.0  # N


class Gripper(ABC):
    """Abstract base class for grippers.

    All gripper implementations should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: GripperConfig):
        self.config = config
        self._is_connected = False
        self._target_width = config.open_width

    @property
    def is_connected(self) -> bool:
        """Whether the gripper is connected."""
        return self._is_connected

    @abstractmethod
    def connect(self, robot: Any) -> None:
        """Connect to the gripper.

        Args:
            robot: Robot instance (type depends on gripper implementation)
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the gripper."""
        pass

    @abstractmethod
    def get_state(self) -> float:
        """Get current gripper width.

        Returns:
            Current gripper width in meters
        """
        pass

    @abstractmethod
    def move(self, width: float) -> None:
        """Move gripper to target width.

        Args:
            width: Target width in meters
        """
        pass

    def clamp_width(self, width: float) -> float:
        """Clamp width to valid range."""
        return max(self.config.close_width, min(self.config.open_width, width))


class NoGripper(Gripper):
    """Dummy gripper implementation when no gripper is attached."""

    def __init__(self, config: GripperConfig | None = None):
        if config is None:
            config = GripperConfig(gripper_type=GripperType.NONE)
        super().__init__(config)

    def connect(self, robot: Any) -> None:
        """No-op for dummy gripper."""
        self._is_connected = True
        logger.info("No gripper configured.")

    def disconnect(self) -> None:
        """No-op for dummy gripper."""
        self._is_connected = False

    def get_state(self) -> float:
        """Return the target width (no actual gripper)."""
        return self._target_width

    def move(self, width: float) -> None:
        """Store target width (no actual gripper)."""
        self._target_width = self.clamp_width(width)


class FlexivGravGripper(Gripper):
    """Flexiv Grav gripper implementation using flexivrdk."""

    def __init__(self, config: FlexivGravGripperConfig):
        super().__init__(config)
        self.config: FlexivGravGripperConfig = config
        self._gripper = None

    def connect(self, robot: Any) -> None:
        """Connect to the Flexiv Grav gripper.

        Args:
            robot: flexivrdk.Robot instance
        """
        try:
            import flexivrdk

            self._gripper = flexivrdk.Gripper(robot)
            self._is_connected = True
            logger.info("Flexiv Grav gripper connected.")
        except Exception as e:
            logger.warning(f"Failed to connect Flexiv Grav gripper: {e}")
            self._gripper = None
            self._is_connected = False

    def disconnect(self) -> None:
        """Disconnect from the gripper."""
        self._gripper = None
        self._is_connected = False
        logger.info("Flexiv Grav gripper disconnected.")

    def get_state(self) -> float:
        """Get current gripper width from hardware.

        Returns:
            Current gripper width in meters
        """
        if self._gripper is None:
            return self._target_width

        try:
            states = self._gripper.states()
            return float(states.width)
        except Exception as e:
            logger.warning(f"Failed to read gripper state: {e}")
            return self._target_width

    def move(self, width: float) -> None:
        """Move gripper to target width.

        Args:
            width: Target width in meters
        """
        if self._gripper is None:
            self._target_width = self.clamp_width(width)
            return

        target = self.clamp_width(width)

        # Only send command if target changed significantly
        if abs(target - self._target_width) > 1e-4:
            self._target_width = target
            try:
                self._gripper.Move(
                    target,
                    self.config.speed,
                    self.config.force,
                )
            except Exception as e:
                logger.warning(f"Failed to move gripper: {e}")


def make_gripper(config: GripperConfig) -> Gripper:
    """Factory function to create gripper instances.

    Args:
        config: Gripper configuration

    Returns:
        Gripper instance matching the configuration
    """
    if config.gripper_type == GripperType.NONE:
        return NoGripper(config)
    elif config.gripper_type == GripperType.FLEXIV_GRAV:
        if not isinstance(config, FlexivGravGripperConfig):
            # Convert base config to FlexivGravGripperConfig
            config = FlexivGravGripperConfig(
                open_width=config.open_width,
                close_width=config.close_width,
            )
        return FlexivGravGripper(config)
    else:
        raise ValueError(f"Unknown gripper type: {config.gripper_type}")

