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

import math
import platform
import time


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


def normalize_quaternion(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float, float]:
    """Normalize quaternion to unit length.
    
    Args:
        qw, qx, qy, qz: Quaternion components
    
    Returns:
        Normalized quaternion as tuple (qw, qx, qy, qz)
    """
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    return (qw / norm, qx / norm, qy / norm, qz / norm)
