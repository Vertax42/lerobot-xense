# Copyright 2025 The HuggingFace & XenseRobotics Inc. team. All rights reserved.
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

import math
import platform
import time

import numpy as np


def busy_wait(seconds):
    if platform.system() == "Darwin" or platform.system() == "Windows":
        # On Mac and Windows, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (roll, pitch, yaw) to quaternion [qw, qx, qy, qz].

    Args:
        roll: Rotation around x-axis in radians
        pitch: Rotation around y-axis in radians
        yaw: Rotation around z-axis in radians

    Returns:
        Tuple of (qw, qx, qy, qz) representing the quaternion
    """
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return (qw, qx, qy, qz)


def normalize_quaternion(q: np.ndarray, input_format: str = "auto") -> np.ndarray:
    """Normalize quaternion and convert to [qw, qx, qy, qz] format (Flexiv convention).

    Args:
        q: Quaternion as numpy array with 4 elements
        input_format: Input quaternion format:
            - "wxyz": [qw, qx, qy, qz] format (Flexiv, scipy)
            - "xyzw": [qx, qy, qz, qw] format (Pico4, ROS, OpenGL)
            - "auto": Auto-detect based on heuristic (default)
                      Heuristic: |w| = |cos(θ/2)| is typically larger for small rotations.
                      Compares first and last elements to determine format.

    Returns:
        Normalized quaternion in [qw, qx, qy, qz] format (Flexiv convention)
    """
    q = np.asarray(q, dtype=np.float32)
    if q.ndim > 1:
        q = q.flatten()
    if len(q) != 4:
        raise ValueError(f"Quaternion must have 4 components, got {len(q)}")

    # Check norm and normalize if needed
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        # Invalid quaternion, return identity
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Skip normalization if already unit quaternion (|norm - 1| < tolerance)
    if abs(norm - 1.0) > 1e-6:
        q = q / norm

    # Determine format and convert to [qw, qx, qy, qz]
    if input_format == "wxyz":
        # Already in [qw, qx, qy, qz] format
        return q.astype(np.float32)
    elif input_format == "xyzw":
        # Convert from [qx, qy, qz, qw] to [qw, qx, qy, qz]
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    elif input_format == "auto":
        # Heuristic: for most rotations, |w| = |cos(θ/2)| tends to be larger
        # Compare first element (if wxyz) vs last element (if xyzw)
        if abs(q[0]) >= abs(q[3]):
            # Likely [qw, qx, qy, qz] format (first element is w)
            return q.astype(np.float32)
        else:
            # Likely [qx, qy, qz, qw] format (last element is w)
            return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown input_format: {input_format}. Use 'wxyz', 'xyzw', or 'auto'.")
