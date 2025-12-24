#!/usr/bin/env python

# Copyright 2025 The Xense Robotics Inc. team. All rights reserved.
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

"""Pose utility functions for coordinate transformations."""

import numpy as np


def xyz_quaternion_to_matrix(x: float, y: float, z: float, 
                              qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert position and quaternion to 4x4 transformation matrix.
    
    Args:
        x, y, z: Position coordinates
        qx, qy, qz, qw: Quaternion components (scalar-last convention)
        
    Returns:
        4x4 transformation matrix
    """
    rot_matrix = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, x],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw, y],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy, z],
        [0, 0, 0, 1],
    ])
    return rot_matrix


def xyz_rpy_to_matrix(x: float, y: float, z: float,
                       roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert position and RPY angles to 4x4 transformation matrix.
    
    Args:
        x, y, z: Position coordinates
        roll, pitch, yaw: Euler angles in radians
        
    Returns:
        4x4 transformation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_matrix = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y],
        [-sp, cp*sr, cp*cr, z],
        [0, 0, 0, 1],
    ])
    return rot_matrix


def matrix_to_xyz_quaternion(matrix: np.ndarray) -> tuple:
    """
    Convert 4x4 transformation matrix to position and quaternion.
    
    Args:
        matrix: 4x4 transformation matrix
        
    Returns:
        Tuple of (x, y, z, qx, qy, qz, qw)
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

    return x, y, z, qx, qy, qz, qw


def quaternion_to_euler(qw: float, qx: float, qy: float, qz: float) -> tuple:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        qw, qx, qy, qz: Quaternion components (scalar-first convention)
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple:
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll, pitch, yaw: Euler angles in radians
        
    Returns:
        Tuple of (qw, qx, qy, qz)
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz

