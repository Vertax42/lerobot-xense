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

"""Pure-serial Xense gripper driver (arm-agnostic, shared across robots).

Uses XenseSerialGripper (from the XGripper submodule, import package
``xgripper``) directly over a USB-serial port. Identified by board SN
(auto-scan) or explicit serial port — no IP / MAC / network.

No ezros / xensesdk stack required, which is a property worth stating because it
is easy to lose: XGripper only regained it in 39c7aad, where a tactile-feedback
controller needed by the TCP gripper was being imported at module scope and so
made the SDK mandatory here too. The tactile sensors that share this gripper's
USB hub are read by the *robot* as XenseTactileCamera entries, not by this
driver.
"""

import time
from threading import Thread

from xgripper import XenseSerialGripper

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger

from ..gripper import Gripper
from .configuration_serial import SerialGripperConfig

# find_port_by_sn is re-exported here for backward compatibility; both it and the
# parity-based find_port_by_side share discovery's scan lock so exact-SN
# lookups and side discovery never interleave on the RS-485 bus.
from .discovery import find_port_by_side, find_port_by_sn

# Log an error only after repeated real communication-health failures.
# Mere status-query timeouts are expected while the MCU is idle and are not errors.
_POLL_FAIL_LOG_THRESHOLD = 20
_POLL_FAIL_REPEAT_INTERVAL = 100
_POLL_FAIL_SLEEP_S = 0.05

# Hybrid initialization: skip motion if already near target, otherwise prefer a
# sync move only before the background poller starts. Once the poller is running,
# avoid blocking sync waits that would contend for the same serial status channel.
_INIT_POSITION_TOLERANCE = 0.02
_INIT_STATUS_TIMEOUT_S = 0.2
_INIT_SYNC_TIMEOUT_S = 3.0


class SerialGripper(Gripper):
    """Wrapper around XenseSerialGripper for use inside BiFlexivRizon4RT.

    Normalized position convention:
        0.0  →  fully closed (SDK position = gripper_min_pos, e.g.  0 mm)
        1.0  →  fully open   (SDK position = gripper_max_pos, e.g. 85 mm)

    Note: XenseSerialGripper uses 0 mm = fully closed, 85 mm = fully open internally.
    Normalized maps directly: normalized * span + min_pos → SDK position.

    Example::

        cfg = SerialGripperConfig(port="/dev/ttyUSB0")
        g = SerialGripper(cfg)
        g.connect()
        g.set_gripper_position(0.5)  # half-closed
        print(g.get_gripper_position())
        g.disconnect()
    """

    config_class = SerialGripperConfig

    def __init__(self, config: SerialGripperConfig):
        super().__init__(config)
        self._config = config
        self._gripper_min_pos = config.gripper_min_pos
        self._gripper_max_pos = config.gripper_max_pos
        self._gripper_v_max = config.gripper_v_max
        self._gripper_f_max = config.gripper_f_max
        self._init_open = config.init_open

        # Resolved at connect() time (explicit port, SN lookup, or side discovery)
        self._port: str = config.port

        label = config.side or config.sn or config.port.split("/")[-1]
        self.logger = get_logger(f"SerialGripper-{label}")
        self._is_connected: bool = False
        self._gripper: XenseSerialGripper | None = None

        # Cached position updated by background poller — avoids blocking get_observation().
        # Pre-seeded from init_open so get_gripper_position() returns a sensible value
        # before the poller thread gets its first valid MCU response (MCU does not
        # respond to status queries while idle, so the first update may be delayed).
        self._cached_position: float = 1.0 if config.init_open else 0.0
        self._poll_thread: Thread | None = None
        self._poll_running: bool = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ── Connection lifecycle ───────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the serial port and start the background receive thread."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Resolve the serial port: explicit port > exact SN scan > parity discovery.
        # The config deliberately allows all three to be unset (a bimanual recipe
        # writes one shared block and the arm stamps `side` per side), so this is
        # where a block that never got an identity has to be caught.
        if not self._config.port and not self._config.sn and not self._config.side:
            raise ValueError(
                f"{self}: cannot resolve a serial port — set one of 'port', 'sn' or "
                "'side' on the gripper config. On a bimanual arm the side is stamped "
                "automatically, so seeing this means the gripper was built standalone."
            )

        if self._config.port:
            self._port = self._config.port
        elif self._config.sn:
            self.logger.info(f"Scanning serial ports for gripper SN={self._config.sn!r}...")
            self._port = find_port_by_sn(
                self._config.sn,
                baudrate=self._config.baudrate,
                device_id=self._config.device_id,
            )
            self.logger.info(f"Found SN={self._config.sn!r} on {self._port}.")
        else:
            self.logger.info(
                f"Auto-discovering {self._config.side} gripper by board-SN parity (odd → left, even → right)..."
            )
            self._port = find_port_by_side(
                self._config.side,
                baudrate=self._config.baudrate,
                device_id=self._config.device_id,
            )
            self.logger.info(f"Discovered {self._config.side} gripper on {self._port}.")

        self.logger.info(
            f"Connecting serial gripper on {self._port} (baud={self._config.baudrate}, id={self._config.device_id})..."
        )
        try:
            self._gripper = XenseSerialGripper(
                port=self._port,
                device_id=self._config.device_id,
                baudrate=self._config.baudrate,
                timeout=self._config.serial_timeout,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open serial gripper on {self._port}: {e}") from e

        self._is_connected = True
        self._cached_position = 1.0 if self._init_open else 0.0
        self.logger.info(f"Serial gripper connected on {self._port}.")

        if self._init_open:
            try:
                self.initialize_gripper_position(
                    1.0,
                    tolerance=_INIT_POSITION_TOLERANCE,
                    status_timeout=_INIT_STATUS_TIMEOUT_S,
                    sync_timeout=_INIT_SYNC_TIMEOUT_S,
                    vmax=self._gripper_v_max,
                    fmax=self._gripper_f_max / 2,
                )
            except Exception as e:
                self.logger.warn(f"Gripper init-open failed (non-fatal): {e}")

        # Start background position poller so get_gripper_position() never blocks
        self._poll_running = True
        self._poll_thread = Thread(target=self._position_poll_loop, daemon=True)
        self._poll_thread.start()

    def _position_poll_loop(self) -> None:
        """Background thread: continuously refresh _cached_position via serial query."""
        span = self._gripper_max_pos - self._gripper_min_pos
        consecutive_failures = 0
        while self._poll_running:
            gripper = self._gripper
            if gripper is None:
                break
            try:
                status = gripper.get_gripper_status(timeout=0.05)
                failure_reason = gripper.get_receive_thread_failure_reason()
                if failure_reason is not None:
                    consecutive_failures += 1
                    self._maybe_log_poll_failure(
                        f"receive thread unhealthy: {failure_reason}",
                        consecutive_failures,
                    )
                    time.sleep(_POLL_FAIL_SLEEP_S)
                    continue

                if consecutive_failures >= _POLL_FAIL_LOG_THRESHOLD:
                    self.logger.info(
                        f"Gripper communication recovered on {self._port} after "
                        f"{consecutive_failures} health-check failures."
                    )
                consecutive_failures = 0

                raw_pos = None if status is None else status.get("position")
                if raw_pos is None:
                    time.sleep(_POLL_FAIL_SLEEP_S)
                    continue

                raw_pos = max(self._gripper_min_pos, min(float(raw_pos), self._gripper_max_pos))
                self._cached_position = (raw_pos - self._gripper_min_pos) / span
            except Exception as e:
                consecutive_failures += 1
                self._maybe_log_poll_failure(
                    f"status poll error: {type(e).__name__}: {e}",
                    consecutive_failures,
                )
                time.sleep(_POLL_FAIL_SLEEP_S)

    def _maybe_log_poll_failure(self, reason: str, consecutive_failures: int) -> None:
        if consecutive_failures == _POLL_FAIL_LOG_THRESHOLD:
            self.logger.error(
                f"Gripper communication unhealthy on {self._port}: {reason} — "
                f"position cache is stale after {consecutive_failures} consecutive failures."
            )
        elif (
            consecutive_failures > _POLL_FAIL_LOG_THRESHOLD
            and (consecutive_failures - _POLL_FAIL_LOG_THRESHOLD) % _POLL_FAIL_REPEAT_INTERVAL == 0
        ):
            self.logger.error(
                f"Gripper communication still unhealthy on {self._port}: {reason} — "
                f"{consecutive_failures} consecutive failures."
            )

    def disconnect(self) -> None:
        """Open gripper fully, stop the background thread, and close the serial port."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Open gripper before disconnecting so it doesn't stay closed
        if self._gripper is not None:
            try:
                self.logger.info("Opening gripper before disconnect...")
                self._gripper.set_position(
                    self._gripper_max_pos,
                    vmax=self._gripper_v_max,
                    fmax=self._gripper_f_max / 2,
                )
            except Exception as e:
                self.logger.warn(f"Gripper open before disconnect failed (non-fatal): {e}")

        self._poll_running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=0.5)
            self._poll_thread = None

        self.logger.info("Disconnecting serial gripper...")
        if self._gripper is not None:
            try:
                self._gripper.release()
            except Exception as e:
                self.logger.debug(f"Error releasing serial gripper: {e}")
            self._gripper = None

        self._is_connected = False
        self.logger.info("Serial gripper disconnected.")

    # ── Position interface ─────────────────────────────────────────────────────

    def get_gripper_position(self) -> float:
        """Return normalized gripper position in [0, 1] from the background cache.

        Returns immediately (non-blocking). The value is refreshed by a background
        poller thread at ~serial-round-trip rate (~10ms), so latency is at most one
        poll cycle behind reality — acceptable for 30 Hz teleoperation.

        Returns:
            0.0 = fully closed, 1.0 = fully open.
        """
        if not self._is_connected:
            return 0.0
        return self._cached_position

    def _position_mm_to_normalized(self, position_mm: float) -> float:
        span = self._gripper_max_pos - self._gripper_min_pos
        clamped = max(self._gripper_min_pos, min(float(position_mm), self._gripper_max_pos))
        return (clamped - self._gripper_min_pos) / span

    def initialize_gripper_position(
        self,
        normalized_pos: float,
        tolerance: float = _INIT_POSITION_TOLERANCE,
        status_timeout: float = _INIT_STATUS_TIMEOUT_S,
        sync_timeout: float = _INIT_SYNC_TIMEOUT_S,
        vmax: float | None = None,
        fmax: float | None = None,
    ) -> None:
        """Hybrid initialization for a target position.

        Behavior:
        - If current position is known and already near target, skip motion.
        - If current position is known from a direct status read before the
          background poller starts, use a sync move.
        - Otherwise fall back to a non-blocking position command.
        """
        if not self._is_connected or self._gripper is None:
            raise DeviceNotConnectedError("Serial gripper is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(f"normalized_pos must be in [0, 1], got {normalized_pos}.")
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}.")

        current_normalized: float | None = None
        current_source = "unavailable"
        poller_running = self._poll_thread is not None and self._poll_thread.is_alive()

        if poller_running:
            current_normalized = self._cached_position
            current_source = "cache"
        else:
            try:
                status = self._gripper.get_gripper_status(timeout=status_timeout)
            except Exception as e:
                self.logger.warn(
                    f"Gripper init status read failed on {self._port}; falling back to non-blocking command: {e}"
                )
                status = None
            raw_pos = None if status is None else status.get("position")
            if raw_pos is not None:
                current_normalized = self._position_mm_to_normalized(raw_pos)
                current_source = "status"
                self._cached_position = current_normalized

        if current_normalized is not None:
            error = abs(current_normalized - normalized_pos)
            if error <= tolerance:
                self.logger.info(
                    f"Gripper already near init target on {self._port} "
                    f"({current_source}: current={current_normalized:.3f}, target={normalized_pos:.3f}); skipping move."
                )
                return

        if current_source == "status":
            self.logger.info(
                f"Initializing gripper on {self._port} with sync move "
                f"(current={current_normalized:.3f}, target={normalized_pos:.3f})..."
            )
            try:
                self.set_gripper_position_sync(
                    normalized_pos,
                    timeout=sync_timeout,
                    vmax=vmax,
                    fmax=fmax,
                )
                self._cached_position = normalized_pos
                self.logger.info("Gripper init sync move completed.")
                return
            except Exception as e:
                self.logger.warn(
                    f"Gripper init sync move failed on {self._port}; falling back to non-blocking command: {e}"
                )

        span = self._gripper_max_pos - self._gripper_min_pos
        target_mm = self._gripper_min_pos + normalized_pos * span
        self._gripper.set_position(
            target_mm,
            vmax=vmax if vmax is not None else self._gripper_v_max,
            fmax=fmax if fmax is not None else self._gripper_f_max,
        )
        action = "open" if normalized_pos >= 0.999 else "close" if normalized_pos <= 0.001 else "position"
        self.logger.info(
            f"Gripper init {action} command sent on {self._port} (non-blocking fallback, source={current_source})."
        )

    def set_gripper_position(self, normalized_pos: float) -> None:
        """Send a position command to the gripper.

        Args:
            normalized_pos: Target position in [0, 1].
                            0.0 = fully closed, 1.0 = fully open.
        """
        if not self._is_connected or self._gripper is None:
            raise DeviceNotConnectedError("Serial gripper is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(f"normalized_pos must be in [0, 1], got {normalized_pos}.")
        span = self._gripper_max_pos - self._gripper_min_pos
        target_mm = self._gripper_min_pos + normalized_pos * span
        self._gripper.set_position(
            target_mm,
            vmax=self._gripper_v_max,
            fmax=self._gripper_f_max,
        )

    def set_gripper_position_sync(
        self,
        normalized_pos: float,
        timeout: float = 10.0,
        vmax: float | None = None,
        fmax: float | None = None,
    ) -> None:
        """Send a position command and block until the gripper reaches the target.

        Args:
            normalized_pos: Target position in [0, 1] (0.0 = closed, 1.0 = open).
            timeout:        Maximum wait time in seconds (default: 10.0).
            vmax:           Override velocity limit mm/s; uses config default if None.
            fmax:           Override force limit N; uses config default if None.
        """
        if not self._is_connected or self._gripper is None:
            raise DeviceNotConnectedError("Serial gripper is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(f"normalized_pos must be in [0, 1], got {normalized_pos}.")
        span = self._gripper_max_pos - self._gripper_min_pos
        target_mm = self._gripper_min_pos + normalized_pos * span
        self._gripper.set_position_sync(
            target_mm,
            vmax=vmax if vmax is not None else self._gripper_v_max,
            fmax=fmax if fmax is not None else self._gripper_f_max,
            timeout=timeout,
        )
