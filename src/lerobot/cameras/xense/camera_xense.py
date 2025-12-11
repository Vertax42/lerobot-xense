# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Provides the XenseTactileCamera class for capturing tactile data from Xense sensors.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from xensesdk import CameraSource

from ..camera import Camera
from .configuration_xense import XenseCameraConfig, XenseOutputType

logger = logging.getLogger(__name__)


class XenseTactileCamera(Camera):
    """
    Manages tactile sensor interactions using Xense SDK for efficient data recording.

    This class provides a high-level interface to connect to, configure, and read
    tactile data from Xense sensors. It supports both synchronous and asynchronous
    data reading with a background thread.

    A XenseTactileCamera instance requires a sensor serial number (e.g., "OG000344").
    The sensor provides various output types including force distribution, force
    resultant, depth maps, and 2D marker tracking.

    Example:
        ```python
        from lerobot.cameras.xense import XenseTactileCamera, XenseCameraConfig, XenseOutputType

        # Basic usage with force sensing
        config = XenseCameraConfig(
            serial_number="OG000344",
            fps=60,
            output_types=[XenseOutputType.FORCE, XenseOutputType.FORCE_RESULTANT]
        )
        sensor = XenseTactileCamera(config)
        sensor.connect()

        # Read data synchronously
        data = sensor.read()
        print(f"Force shape: {data['force'].shape}")
        print(f"Force resultant: {data['force_resultant']}")

        # Read data asynchronously
        async_data = sensor.async_read()

        # When done, properly disconnect
        sensor.disconnect()
        ```
    """

    def __init__(self, config: XenseCameraConfig):
        """
        Initializes the XenseTactileCamera instance.

        Args:
            config: The configuration settings for the Xense sensor.
        """
        super().__init__(config)

        self.config = config
        self.serial_number = config.serial_number
        self.output_types = config.output_types
        self.warmup_s = config.warmup_s
        self.rectify_size = config.rectify_size
        self.raw_size = config.raw_size

        self.sensor = None

        # Threading for async read
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_data: dict[str, np.ndarray] | None = None
        self.new_frame_event: Event = Event()

        # Import xensesdk here to avoid import errors if not installed
        try:
            from xensesdk import Sensor

            self._Sensor = Sensor
        except ImportError as e:
            raise ImportError(
                "xensesdk is required for XenseTactileCamera. "
                "Please install it according to the manufacturer's instructions."
            ) from e

        # Pre-build sensor output types list for better performance
        # This avoids reconstructing the mapping on every read() call
        self._sensor_output_types_cache = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        """Checks if the sensor is currently connected."""
        return self.sensor is not None

    def connect(self, warmup: bool = True):
        """
        Connects to the Xense sensor specified in the configuration.

        Initializes the Xense Sensor object and performs initial checks.

        Raises:
            DeviceAlreadyConnectedError: If the sensor is already connected.
            ConnectionError: If the sensor fails to connect.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            # Use default OpenCV backend (no api parameter = CV2_V4L2)
            self.sensor = self._Sensor.create(
                self.serial_number,
                api=CameraSource.CV2_V4L2,
                rectify_size=self.rectify_size,
                raw_size=self.raw_size,
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to {self}. Error: {e}. "
                "Make sure the sensor is plugged in and the serial number is correct."
            ) from e

        if warmup:
            # time.sleep(2)
            # Start background thread first for async_read
            # Do warmup reads to stabilize the sensor
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    self.read()
                except Exception:
                    pass  # Ignore errors during warmup
                time.sleep(0.1)
            self._start_read_thread()

        logger.info(f"{self} connected with CV2_V4L2 API (OpenCV backend)")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Xense sensors connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'serial_number', and other info.
        """
        try:
            from xensesdk import Sensor

            # Get available devices - returns dict like {'OG000352': 10, 'OG000344': 8}
            # Key is serial number, value is cam_id (video device index)
            devices = Sensor.scanSerialNumber()

            found_sensors = []
            for serial_number, cam_id in devices.items():
                sensor_info = {
                    "name": f"Xense Tactile Sensor {serial_number}",
                    "type": "Xense",
                    "serial_number": serial_number,
                    "cam_id": cam_id,
                }
                found_sensors.append(sensor_info)

            return found_sensors

        except ImportError:
            logger.warning(
                "xensesdk not installed. Cannot detect Xense sensors. "
                "Please install xensesdk to use Xense tactile sensors."
            )
            return []
        except Exception as e:
            logger.error(f"Error detecting Xense sensors: {e}")
            return []

    def _read_sensor_data(self) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Internal method to read data from the sensor based on configured output types.

        Returns:
            np.ndarray if single output type, or tuple of np.ndarray if multiple output types.

        Raises:
            DeviceNotConnectedError: If the sensor is not connected.
            RuntimeError: If reading from the sensor fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # Build sensor output types list (cached for performance)
            if self._sensor_output_types_cache is None:
                # Map XenseOutputType to Sensor.OutputType
                # Note: SDK uses CamelCase for OutputType attributes (e.g., Force, ForceResultant)
                output_type_mapping = {
                    XenseOutputType.RECTIFY: self._Sensor.OutputType.Rectify,
                    XenseOutputType.DIFFERENCE: self._Sensor.OutputType.Difference,
                    XenseOutputType.DEPTH: self._Sensor.OutputType.Depth,
                    XenseOutputType.MARKER_2D: self._Sensor.OutputType.Marker2D,
                    XenseOutputType.FORCE: self._Sensor.OutputType.Force,
                    XenseOutputType.FORCE_NORM: self._Sensor.OutputType.ForceNorm,
                    XenseOutputType.FORCE_RESULTANT: self._Sensor.OutputType.ForceResultant,
                    XenseOutputType.MESH_3D: self._Sensor.OutputType.Mesh3D,
                    XenseOutputType.MESH_3D_INIT: self._Sensor.OutputType.Mesh3DInit,
                    XenseOutputType.MESH_3D_FLOW: self._Sensor.OutputType.Mesh3DFlow,
                }

                # Build list of sensor output types to request (cache it)
                self._sensor_output_types_cache = [
                    output_type_mapping[output_type]
                    for output_type in self.output_types
                ]

            # Call selectSensorInfo with cached sensor output types
            # Returns: single np.ndarray if one arg, or tuple of np.ndarray if multiple args
            results = self.sensor.selectSensorInfo(*self._sensor_output_types_cache)

            # image_outputs definition
            image_outputs = {
                XenseOutputType.RECTIFY,
                XenseOutputType.DIFFERENCE,
                XenseOutputType.DEPTH,
            }

            # DEBUG: Check if output type is in image_outputs
            # print(f"DEBUG: output_type={self.output_types[0]}, in_image_outputs={self.output_types[0] in image_outputs}")

            # xensesdk may return either tuple or list when multiple outputs are requested.
            # Accept both to avoid treating a list of heterogeneous arrays as a single output,
            # which triggers numpy's "inhomogeneous shape" ValueError.
            if isinstance(results, (tuple, list)):
                processed_results: list[np.ndarray] = []
                for i, output_type in enumerate(self.output_types):
                    data = np.asarray(results[i])

                    if output_type in image_outputs and data.ndim >= 2:
                        # Transpose only image-like outputs; keep force/marker/resultant unchanged
                        data = np.transpose(
                            data, (1, 0) + tuple(range(2, data.ndim))
                        )

                    processed_results.append(data)

                return tuple(processed_results)
            else:
                # single output type
                results = np.asarray(results)
                if self.output_types[0] in image_outputs and results.ndim >= 2:
                    results = np.transpose(
                        results, (1, 0) + tuple(range(2, results.ndim))
                    )
                return results

        except Exception as e:
            raise RuntimeError(f"{self} failed to read sensor data: {e}") from e

    def read(self, color_mode=None) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Reads tactile data synchronously from the sensor.

        This is a blocking call. It waits for the next available data from the sensor.

        Args:
            color_mode: Not used for Xense sensors, kept for API compatibility.

        Returns:
            np.ndarray if single output type configured, or tuple of np.ndarray if multiple.
            For example:
            - Single: array(35,20,3) for FORCE only
            - Multiple: (array(35,20,3), array(6,)) for FORCE and FORCE_RESULTANT

        Raises:
            DeviceNotConnectedError: If the sensor is not connected.
            RuntimeError: If reading from the sensor fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        data = self._read_sensor_data()

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return data

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads sensor data
        2. Stores result in latest_data (thread-safe)
        3. Sets new_frame_event to notify listeners
        4. Sleeps to maintain target FPS

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        target_loop_time = 1.0 / self.fps if self.fps else 0

        while not self.stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                data = self.read()

                with self.frame_lock:
                    self.latest_data = data
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(
                    f"Error reading data in background thread for {self}: {e}"
                )

            # Sleep to maintain target FPS
            if target_loop_time > 0:
                elapsed = time.perf_counter() - loop_start
                sleep_time = target_loop_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(
        self, timeout_ms: float = 200
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Reads the latest available data asynchronously.

        This method retrieves the most recent data captured by the background
        read thread. It returns immediately with the latest cached data, or waits
        up to timeout_ms if no data is available yet (e.g., first call after connect).

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for data
                to become available on first call. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray if single output type configured, or tuple of np.ndarray if multiple.
            For example:
            - Single: array(35,20,3) for FORCE only
            - Multiple: (array(35,20,3), array(6,)) for FORCE and FORCE_RESULTANT

        Raises:
            DeviceNotConnectedError: If the sensor is not connected.
            TimeoutError: If no data becomes available within the specified timeout.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        # Try to get data immediately (non-blocking)
        with self.frame_lock:
            data = self.latest_data

        # If no data yet (first call), wait for it
        if data is None:
            if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
                thread_alive = self.thread is not None and self.thread.is_alive()
                raise TimeoutError(
                    f"Timed out waiting for data from sensor {self} after {timeout_ms} ms. "
                    f"Read thread alive: {thread_alive}."
                )

            with self.frame_lock:
                data = self.latest_data

            if data is None:
                raise RuntimeError(
                    f"Internal error: Event set but no data available for {self}."
                )

        return data

    def disconnect(self):
        """
        Disconnects from the sensor and cleans up resources.

        Stops the background read thread (if running) and releases the sensor.

        Raises:
            DeviceNotConnectedError: If the sensor is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.sensor is not None:
            try:
                self.sensor.release()
            except Exception as e:
                logger.warning(f"Error releasing {self}: {e}")
            self.sensor = None

        logger.info(f"{self} disconnected.")
