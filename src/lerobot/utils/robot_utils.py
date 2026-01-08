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

import platform
import time

import numpy as np
import spdlog

# Default spdlog pattern with color markers: [HH:MM:SS] [logger_name] [level] message
# %^ and %$ mark the colored region (level name will be colored)
SPDLOG_PATTERN = "[%H:%M:%S] [%n] [%^%l%$] %v"


def get_logger(name: str) -> spdlog.ConsoleLogger:
    """Create a spdlog ConsoleLogger with unified format and colors.

    Args:
        name: Logger name

    Returns:
        Configured spdlog.ConsoleLogger instance with colored output
    """
    logger = spdlog.ConsoleLogger(name, colored=True, multithreaded=True)
    logger.set_pattern(SPDLOG_PATTERN)
    logger.set_level(spdlog.LogLevel.INFO)
    return logger


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


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion [qw, qx, qy, qz].

    Uses ZYX intrinsic rotation order (yaw → pitch → roll), which is:
    - First rotate around Z-axis by yaw
    - Then rotate around Y-axis by pitch
    - Finally rotate around X-axis by roll

    This is consistent with Flexiv SDK convention and aerospace/aviation standard.

    Args:
        roll: Rotation around x-axis in radians
        pitch: Rotation around y-axis in radians
        yaw: Rotation around z-axis in radians

    Returns:
        np.ndarray of shape (4,) in [qw, qx, qy, qz] format.
    """
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    return np.array(
        [
            cr * cp * cy + sr * sp * sy,  # qw
            sr * cp * cy - cr * sp * sy,  # qx
            cr * sp * cy + sr * cp * sy,  # qy
            cr * cp * sy - sr * sp * cy,  # qz
        ],
        dtype=np.float32,
    )


def quaternion_to_euler(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw).

    Uses ZYX intrinsic rotation order, consistent with Flexiv SDK and aerospace standard.
    This is the inverse of euler_to_quaternion().

    Note: Gimbal lock occurs when pitch ≈ ±90°, causing roll and yaw to become coupled.

    Args:
        qw: Quaternion scalar component
        qx: Quaternion x component
        qy: Quaternion y component
        qz: Quaternion z component

    Returns:
        np.ndarray of shape (3,) in [roll, pitch, yaw] order (radians):
        - roll: Rotation around x-axis, range [-π, π]
        - pitch: Rotation around y-axis, range [-π/2, π/2]
        - yaw: Rotation around z-axis, range [-π, π]
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation) with gimbal lock handling
    sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions q1 * q2.

    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]

    Returns:
        np.ndarray of shape (4,) representing q1 * q2 in [qw, qx, qy, qz] format.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


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
        # Invalid quaternion, return identity in input_format
        if input_format == "wxyz":
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif input_format == "xyzw":
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            raise ValueError(
                f"Unknown input_format: {input_format}. Use 'wxyz' or 'xyzw'."
            )

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


# =============================================================================
# 6D Rotation Representation (Continuous Rotation Representation)
# Reference: "On the Continuity of Rotation Representations in Neural Networks"
#            Zhou et al., CVPR 2019
#
# 6D representation uses the first two columns of a rotation matrix.
# Advantages over Euler angles and quaternions:
#   - Continuous: No discontinuities at ±180° boundaries (unlike Euler angles)
#   - No double-cover: Unlike quaternions where q and -q represent the same rotation
#   - Better for neural network learning in robotics applications
# =============================================================================


def quaternion_to_rotation_6d(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion to 6D rotation representation.

    The 6D representation consists of the first two columns of the rotation matrix,
    which can be used to uniquely reconstruct the full rotation matrix via
    Gram-Schmidt orthogonalization.

    Args:
        qw: Quaternion scalar component
        qx: Quaternion x component
        qy: Quaternion y component
        qz: Quaternion z component

    Returns:
        np.ndarray of shape (6,) containing [r1, r2, r3, r4, r5, r6] where:
        - [r1, r2, r3] is the first column of the rotation matrix
        - [r4, r5, r6] is the second column of the rotation matrix
    """
    # Rotation matrix from quaternion
    # First column
    r1 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r2 = 2.0 * (qx * qy + qz * qw)
    r3 = 2.0 * (qx * qz - qy * qw)

    # Second column
    r4 = 2.0 * (qx * qy - qz * qw)
    r5 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r6 = 2.0 * (qy * qz + qx * qw)

    return np.array([r1, r2, r3, r4, r5, r6], dtype=np.float32)


def rotation_6d_to_quaternion(
    r6d: np.ndarray, ensure_positive_w: bool = True
) -> np.ndarray:
    """Convert 6D rotation representation to quaternion.

    Uses Gram-Schmidt orthogonalization to reconstruct the rotation matrix
    from the 6D representation, then converts to quaternion.

    Args:
        r6d: 6D rotation representation [r1, r2, r3, r4, r5, r6]
        ensure_positive_w: If True, ensure qw >= 0 for consistent output.
                          This doesn't change the rotation (q and -q are equivalent).

    Returns:
        np.ndarray of shape (4,) in [qw, qx, qy, qz] format
    """
    r6d = np.asarray(r6d, dtype=np.float64)  # Use float64 for numerical stability
    if r6d.shape != (6,):
        raise ValueError(f"Expected r6d array of shape (6,), got {r6d.shape}")

    # Extract the two column vectors
    a1 = r6d[:3]
    a2 = r6d[3:6]

    # Gram-Schmidt orthogonalization
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    # Construct rotation matrix (columns are b1, b2, b3)
    rot_matrix = np.column_stack([b1, b2, b3])

    # Convert rotation matrix to quaternion using Shepperd's method
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

    q = np.array([qw, qx, qy, qz], dtype=np.float32)

    # Normalize
    q = q / np.linalg.norm(q)

    # Ensure qw >= 0 for consistent output (q and -q represent the same rotation)
    if ensure_positive_w and q[0] < 0:
        q = -q

    return q


def pose7d_to_pose9d(pose: np.ndarray, input_format: str = "wxyz") -> np.ndarray:
    """Convert 7D pose (position + quaternion) to 9D pose (position + 6D rotation).

    This conversion is useful for neural network training as 6D rotation
    representation is continuous and avoids discontinuities present in
    Euler angles and the double-cover issue of quaternions.

    Args:
        pose: 7D array containing position and quaternion.
              Format depends on input_format:
              - "wxyz": [x, y, z, qw, qx, qy, qz] (scalar-first, Flexiv convention)
              - "xyzw": [x, y, z, qx, qy, qz, qw] (scalar-last, ROS convention)
        input_format: Quaternion format in the input pose.

    Returns:
        np.ndarray of shape (9,): [x, y, z, r1, r2, r3, r4, r5, r6]
        where r1-r6 is the 6D rotation representation.
    """
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose array of shape (7,), got {pose.shape}")

    x, y, z = pose[0], pose[1], pose[2]

    if input_format == "wxyz":
        qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
    elif input_format == "xyzw":
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    else:
        raise ValueError(
            f"Unknown input_format: {input_format}. Expected 'wxyz' or 'xyzw'."
        )

    r6d = quaternion_to_rotation_6d(qw, qx, qy, qz)

    return np.concatenate([[x, y, z], r6d]).astype(np.float32)


def pose9d_to_pose7d(
    pose: np.ndarray, output_format: str = "wxyz", ensure_positive_w: bool = True
) -> np.ndarray:
    """Convert 9D pose (position + 6D rotation) to 7D pose (position + quaternion).

    This is the inverse of pose7d_to_pose9d(), used to convert neural network
    outputs back to quaternion format for robot control.

    Args:
        pose: 9D array [x, y, z, r1, r2, r3, r4, r5, r6]
              where r1-r6 is the 6D rotation representation.
        output_format: Quaternion format for output:
              - "wxyz": [x, y, z, qw, qx, qy, qz] (scalar-first, Flexiv convention)
              - "xyzw": [x, y, z, qx, qy, qz, qw] (scalar-last, ROS convention)
        ensure_positive_w: If True, ensure qw >= 0 for consistent output.

    Returns:
        np.ndarray of shape (7,) containing position and quaternion.
    """
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (9,):
        raise ValueError(f"Expected pose array of shape (9,), got {pose.shape}")

    x, y, z = pose[0], pose[1], pose[2]
    r6d = pose[3:9]

    quat = rotation_6d_to_quaternion(r6d, ensure_positive_w=ensure_positive_w)
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]

    if output_format == "wxyz":
        return np.array([x, y, z, qw, qx, qy, qz], dtype=np.float32)
    elif output_format == "xyzw":
        return np.array([x, y, z, qx, qy, qz, qw], dtype=np.float32)
    else:
        raise ValueError(
            f"Unknown output_format: {output_format}. Expected 'wxyz' or 'xyzw'."
        )
