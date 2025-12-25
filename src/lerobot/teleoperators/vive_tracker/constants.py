import numpy as np
from lerobot.utils.robot_utils import quaternion_to_matrix

# Vive Tracker Constants
VIVE2EE_POS = np.array([0.0, 0.0, 0.16])
VIVE2EE_QUAT_WXYZ = np.array([0.676, -0.207, -0.207, -0.676])  # [qw, qx, qy, qz]
VIVE2EE_MATRIX = quaternion_to_matrix(
    np.concatenate([VIVE2EE_POS, VIVE2EE_QUAT_WXYZ]), input_format="wxyz"
)

EE_INIT_POS = np.array([0.693307, -0.114902, 0.14589])
EE_INIT_QUAT_WXYZ = np.array(
    [0.004567, 0.003238, 0.999984, 0.001246]
)  # [qw, qx, qy, qz]
EE_INIT_MATRIX = quaternion_to_matrix(
    np.concatenate([EE_INIT_POS, EE_INIT_QUAT_WXYZ]), input_format="wxyz"
)
