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


def xyz_rpy_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert position and RPY angles to 4x4 transformation matrix.

    Args:
        pose: 6D array [x, y, z, roll, pitch, yaw]
              - x, y, z: Position coordinates
              - roll, pitch, yaw: Euler angles in radians

    Returns:
        4x4 transformation matrix
    """
    if pose.shape != (6,):
        raise ValueError(f"Expected pose array of shape (6,), got {pose.shape}")

    x, y, z = pose[0], pose[1], pose[2]
    roll, pitch, yaw = pose[3], pose[4], pose[5]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_matrix = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
            [-sp, cp * sr, cp * cr, z],
            [0, 0, 0, 1],
        ]
    )
    return rot_matrix


def quaternion_to_matrix(
    pose: np.ndarray,
    input_format: str = "wxyz",
) -> np.ndarray:
    """
    Convert position and quaternion to 4x4 transformation matrix.

    Args:
        pose: 7D array containing position and quaternion.
              Format depends on input_format parameter:
              - "xyzw": [x, y, z, qx, qy, qz, qw] (scalar-last)
              - "wxyz": [x, y, z, qw, qx, qy, qz] (scalar-first)
        input_format: Quaternion format, either "xyzw" (scalar-last) or "wxyz" (scalar-first).
                      Default is "xyzw".

    Returns:
        4x4 transformation matrix
    """
    if pose.shape != (7,):
        raise ValueError(f"Expected pose array of shape (7,), got {pose.shape}")

    x, y, z = pose[0], pose[1], pose[2]

    if input_format == "xyzw":
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    elif input_format == "wxyz":
        qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
    else:
        raise ValueError(
            f"Unknown input_format: {input_format}. Expected 'xyzw' or 'wxyz'."
        )

    rot_matrix = np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
                x,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
                y,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
                z,
            ],
            [0, 0, 0, 1],
        ]
    )
    return rot_matrix


def matrix_to_pose7d(matrix: np.ndarray, output_format: str = "wxyz") -> np.ndarray:
    """
    Convert 4x4 transformation matrix to 7D pose [x, y, z, qw, qx, qy, qz].

    Args:
        matrix: 4x4 transformation matrix
        output_format: Quaternion output format:
            - "xyzw": [x, y, z, qx, qy, qz, qw] (scalar-last)
            - "wxyz": [x, y, z, qw, qx, qy, qz] (scalar-first)
            Default is "wxyz".

    Returns:
        7D array containing position and quaternion
    """
    # Extract position
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]

    # Extract rotation matrix
    rot_matrix = matrix[:3, :3]

    # Calculate quaternion using Shepperd's method
    trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
        qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
        qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
    elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
        qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
        qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
    elif rot_matrix[1, 1] > rot_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
        qw = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
        qx = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
        qw = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
        qx = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        qy = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        qz = 0.25 * s

    if output_format == "xyzw":
        return np.array([x, y, z, qx, qy, qz, qw])
    elif output_format == "wxyz":
        return np.array([x, y, z, qw, qx, qy, qz])
    else:
        raise ValueError(
            f"Unknown output_format: {output_format}. Use 'xyzw' or 'wxyz'."
        )


def euler_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
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


def slerp_quaternion(
    q1: np.ndarray, q2: np.ndarray, t: float, input_format: str = "wxyz"
) -> np.ndarray:
    """Spherical Linear Interpolation (SLERP) between two quaternions.

    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]
        t: Interpolation factor [0, 1], where 0 returns q1 and 1 returns q2
        input_format: Input quaternion format:
            - "wxyz": [qw, qx, qy, qz] format (Flexiv, scipy)
            - "xyzw": [qx, qy, qz, qw] format (Pico4, ROS, OpenGL)
            Default is "wxyz".

    Returns:
        Interpolated quaternion [qw, qx, qy, qz]
    """
    q1 = normalize_quaternion(q1, input_format=input_format)
    q2 = normalize_quaternion(q2, input_format=input_format)

    dot = np.dot(q1, q2)

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if abs(dot) > 0.9995:
        result = q1 + t * (q2 - q1)
        return normalize_quaternion(result, input_format=input_format)

    theta = np.arccos(abs(dot))
    sin_theta = np.sin(theta)

    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    result = w1 * q1 + w2 * q2

    return normalize_quaternion(result, input_format=input_format)


def normalize_quaternion(q: np.ndarray, input_format: str = "wxyz") -> np.ndarray:
    """Normalize quaternion and convert to [qw, qx, qy, qz] format (Flexiv convention).

    Args:
        q: Quaternion as numpy array with 4 elements
        input_format: Input quaternion format:
            - "wxyz": [qw, qx, qy, qz] format (Flexiv, scipy)
            - "xyzw": [qx, qy, qz, qw] format (Pico4, ROS, OpenGL)

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

    # Convert to [qw, qx, qy, qz] format
    if input_format == "wxyz":
        # Already in [qw, qx, qy, qz] format
        return q.astype(np.float32)
    elif input_format == "xyzw":
        # Convert from [qx, qy, qz, qw] to [qw, qx, qy, qz]
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown input_format: {input_format}. Use 'wxyz' or 'xyzw'.")
