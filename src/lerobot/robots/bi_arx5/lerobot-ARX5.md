# ARX5 SDK
conda install ros-humble-kdl-parser ros-humble-ament-cmake cxx-compiler cmake ninja orocos-kdl eigen boost spdlog pybind11 numpy click pyzmq pynput pip conda-forge::soem=1.4.0 -c robostack-staging -c conda-forge

pip install atomics spnav@https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz pyrealsense2

    sudo setcap cap_sys_nice=ep $(readlink -f $CONDA_PREFIX/bin/python)

# for openpi-client compatibility
<!-- pip install opencv-python==4.9.0.80 4.10.0.84
pip install opencv-python-headless==4.9.0.80 -->
<!-- pip install numpy==2.2.6 -, 1.26.4 -->

sudo setcap cap_sys_nice+ep $(readlink -f $(which python))

# C++ ABI version issue
/home/vertax/miniconda3/envs/${CONDA_PREFIX}/lib/${python_version}/site-packages/sitecustomize.py
```python
"""
Sitecustomize for conda environment.

This file is automatically executed when Python starts.
It preloads the conda environment's libstdc++.so.6 to ensure C++ extensions
compiled with GCC 14.3.0 can find the required CXXABI_1.3.15 symbols.
"""
import os
import ctypes

conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    libstdcxx_path = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
    if os.path.exists(libstdcxx_path):
        try:
            # Preload with RTLD_GLOBAL so all subsequently loaded modules can use it
            ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            # Silently fail if preloading doesn't work
            pass
```

# ARX_X5 CAN Bus Diagram

```
┌─────────────────┐    CAN Bus    ┌─────────────────┐
│   Controller    │◄─────────────►│  Motor ID: 1    │ (Joint 1)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 2    │ (Joint 2)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 4    │ (Joint 3)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 5    │ (Joint 4)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 6    │ (Joint 5)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 7    │ (Joint 6)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 8    │ (Gripper)
└─────────────────┘               └─────────────────┘
```

# default configurations X5 in ` include/app/config.h `

```cpp
// joint_names: [0: joint1, 1: joint2, 2: joint3, 3: joint4, 4: joint5, 5: joint6]
// motors: [0: EC_A4310, 1: EC_A4310, 2: EC_A4310, 3: DM_J4310, 4: DM_J4310, 5: DM_J4310]
RobotConfigFactory()
    {
        configurations["X5"] = std::make_shared<RobotConfig>(
            "X5",                                                          // robot_model
            (VecDoF(6) << -3.14, -0.05, -0.1, -1.6, -1.57, -2).finished(), // joint_pos_min
            (VecDoF(6) << 2.618, 3.50, 3.20, 1.55, 1.57, 2).finished(),    // joint_pos_max
            (VecDoF(6) << 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),        // joint_vel_max
            (VecDoF(6) << 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),  // joint_torque_max
            (Pose6d() << 0.6, 0.6, 0.6, 1.8, 1.8, 1.8).finished(),         // ee_vel_max
            0.3,                                                           // gripper_vel_max
            1.5,                                                           // gripper_torque_max
            0.088,                                                         // gripper_width
            5.03,                                                          // gripper_open_readout
            6,                                                             // joint_dof
            std::vector<int>{1, 2, 4, 5, 6, 7},                            // motor_id
            std::vector<MotorType>{MotorType::EC_A4310, MotorType::EC_A4310, MotorType::EC_A4310, MotorType::DM_J4310,
                                   MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                                // gripper_motor_id
            MotorType::DM_J4310,                                              // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                   // gravity_vector
            "base_link",                                                      // base_link_name
            "eef_link",                                                       // eef_link_name
            std::string(SDK_ROOT) + "/models/X5.urdf"                         // urdf_path
        );
    }

ControllerConfigFactory()
    {
        configurations["joint_controller_6"] = std::make_shared<ControllerConfig>(
            "joint_controller",                                           // controller_type
            (VecDoF(6) << 80.0, 70.0, 70.0, 30.0, 30.0, 20.0).finished(), // default_kp
            (VecDoF(6) << 2.0, 2.0, 2.0, 1.0, 1.0, 0.7).finished(),       // default_kd
            2.0,                                                          // default_gripper_kp
            0.1,                                                          // default_gripper_kd
            20,                                                           // over_current_cnt_max
            0.002,                                                        // controller_dt
            true,                                                         // gravity_compensation
            true,                                                         // background_send_recv
            true,                                                         // shutdown_to_passive
            "linear",                                                     // interpolation_method
            0.0                                                           // default_preview_time
        );
    }
```