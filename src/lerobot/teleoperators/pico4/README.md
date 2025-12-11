### PICO Device JSON Data Description

Currently, there are five types of tracking data: head, controller, hand gesture, full-body motion capture, and independent tracking. All data will be sent as a JSON string with the following basic structure.

```json
{"functionName":"Tracking","value":"{"Head":"Head","Controller":"Controller","Hand":"Hand","Body":"Body Motion Capture","Motion":"Tracker Independent Tracking","timeStampNs":"Timestamp","Input":"Input method 0 head 1 controller 2 gesture"}"}
```

Field description

| Key         | value type | Description                                                             |
| ----------- | ---------- | ----------------------------------------------------------------------- |
| function    | string     | Tracking indicates the data is tracking data                            |
| value       | Json       | Contains specific data                                                  |
| Head        | Json       | Headset tracking data                                                   |
| Controller  | Json       | Controller tracking data                                                |
| Hand        | Json       | Hand gesture data                                                       |
| Body        | Json       | Full-body motion capture data                                           |
| Motion      | Json       | Tracker independent tracking data                                       |
| timeStampNs | int64      | Timestamp when the data was obtained, Unix (nanoseconds)                |
| Input       | int        | Current input method, 0 head input, 1 controller input, 2 gesture input |

* If the control panel does not enable tracking for the corresponding part, the corresponding key will not exist.

* value is the actual data string. When using, please convert the string to JSON and remove "\\".

```json
JsonData data = JsonMapper.ToObject(HeadJson);
string valueStr = data["value"].ToString().Replace("\\", "");
JsonData valueJson = JsonMapper.ToObject(valueStr);
```

**Pose**: A string representing seven float data for pose, separated by commas. The first three floats represent position (x, y, z), and the last four floats represent rotation (quaternion: x, y, z, w);

**Coordinate System**: Right-handed coordinate system (X right, Y up, Z in), the origin is set as the head position when the application starts. The following figure marks the position and orientation of the Head point. 

#### 1. **Headset Pose:**

```json
{"pose":"0.0,0.0,0.0,0.0,-0.0,0.0,0.0","status":3,"timeStampNs":1732613222842776064,"handMode":0}"
```



| key         | type                      | description                                                          |
| ----------- | ------------------------- | -------------------------------------------------------------------- |
| pose        | seven float               | Pose (right-handed, X right, Y up, Z in)                             |
| status      | int                       | Indicates confidence (0 not reliable, 1 reliable)                    |
| handMode    | Current hand gesture type | 0 not enabled, 1 controller enabled, 2 hand gesture tracking enabled |
| timeStampNs | int64                     | Unix timestamp (nanoseconds)                                         |

#### 2. **Controller**

```json
{"axisX":0.0,"axisY":0.0,"grip":0.0,"trigger":0.0,"primaryButton":false,"secondaryButton":false,"menuButton":false,"pose":"0.0,0.0,0.0,0.0,0.0,0.0"},"right":{"axisX":0.0,"axisY":0.0,"grip":0.0,"trigger":0.0,"primaryButton":false,"secondaryButton":false,"menuButton":false,"pose":"0.0,0.0,0.0,0.0,0.0,0.0,0.0"},"timeStampNs":1732613438765715200}"
```

* left and right represent the left and right controllers respectively

* pose: controller pose (left-handed, X right, Y up, Z in), with the same orientation system as the head pose. 

* Controller buttons

| **Key**         | **Type** | **Left Controller**          | **Right Controller**     |
| --------------- | -------- | ---------------------------- | ------------------------ |
| menuButton      | bool     | Menu button                  | Screenshot/Record button |
| trigger         | float    | Trigger button               | Trigger button           |
| grip            | float    | Grip button                  | Grip button              |
| axisX, axisY    | float    | Joystick                     | Joystick                 |
| axisClick       | bool     | Joystick click or press      | Joystick click or press  |
| primaryButton   | bool     | X                            | A                        |
| secondaryButton | bool     | Y                            | B                        |
| timeStampNs     | int64    | Unix timestamp (nanoseconds) |                          |

#### 3. **Hand Gesture**

```json
"leftHand":{"isActive":0,"count":26,"scale":1.0,"timeStampNs":1732613438765715200,"HandJointLocations":[{"p":"0,0,0,0,0,0,0","s":0.0,"r":0.0}, ...]},"rightHand":{"isActive":0,"count":26,"scale":1.0,"HandJointLocations":[{"p":"0,0,0,0,0,0,0","s":0.0,"r":0.0}, ...]}
```

* leftHand and rightHand represent left and right hand data respectively

| **key**            | **Type** | **Description**                                                        |
| ------------------ | -------- | ---------------------------------------------------------------------- |
| isActive           | int      | Hand tracking quality (0 low, 1 high)                                  |
| count              | int      | Number of finger joints (fixed = 26)                                   |
| scale              | float    | Hand scale                                                             |
| HandJointLocations | Array    | Array of joint pose data, length = count (see joint description below) |
| timeStampNs        | int64    | Unix timestamp (nanoseconds)                                           |

**Finger Joint Data**

| **key** | **Type**    | **Description**                                                                                                                                                                  |
| ------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| p       | seven float | Pose (left-handed, X right, Y up, Z in)                                                                                                                                          |
| s       | ulong       | Hand joint tracking status, currently only four values: OrientationValid = 0x00000001, PositionValid = 0x00000002, OrientationTracked = 0x00000004, PositionTracked = 0x00000008 |
| r       | float       | Joint radius                                                                                                                                                                     |

**Hand Joint Description**

Below is the description of the 26 finger joints in the HandJointLocations array.


The figure below shows finger joint locations in Unity coordinate system (x right, y up, z out). In the actual data, z-axis follows the z-in direction, same as controller pose and head pose.

![finger_joints](https://github.com/user-attachments/assets/47f1e10d-9e78-4297-a110-a0254b100908)


|     |                     |                           |
| --- | ------------------- | ------------------------- |
| 0   | Palm                | Palm center               |
| 1   | Wrist               | Wrist joint               |
| 2   | Thumb_metacarpal    | Thumb metacarpal joint    |
| 3   | Thumb_proximal      | Thumb proximal joint      |
| 4   | Thumb_distal        | Thumb distal joint        |
| 5   | Thumb_tip           | Thumb tip joint           |
| 6   | Index_metacarpal    | Index metacarpal joint    |
| 7   | Index_proximal      | Index proximal joint      |
| 8   | Index_intermediate  | Index intermediate joint  |
| 9   | Index_distal        | Index distal joint        |
| 10  | Index_tip           | Index tip joint           |
| 11  | Middle_metacarpal   | Middle metacarpal joint   |
| 12  | Middle_proximal     | Middle proximal joint     |
| 13  | Middle_intermediate | Middle intermediate joint |
| 14  | Middle_distal       | Middle distal joint       |
| 15  | Middle_tip          | Middle tip joint          |
| 16  | Ring_metacarpal     | Ring metacarpal joint     |
| 17  | Ring_proximal       | Ring proximal joint       |
| 18  | Ring_intermediate   | Ring intermediate joint   |
| 19  | Ring_distal         | Ring distal joint         |
| 20  | Ring_tip            | Ring tip joint            |
| 21  | Little_metacarpal   | Little metacarpal joint   |
| 22  | Little_proximal     | Little proximal joint     |
| 23  | Little_intermediate | Little intermediate joint |
| 24  | Little_distal       | Little distal joint       |
| 25  | Little_tip          | Little tip joint          |

#### 4. Full-body Motion Capture Tracking


Full-body motion capture requires additional Pico Swift devices (at least two) and proper adaptation and calibration in the Pico headset.

![](images/image-25.png)

> Note: Each time the headset is activated, calibration is required.

**Human Joint Reference**

The full-body motion capture function of the PICO SDK supports tracking the 24 human joints shown in the figure below.
<div align="center">
  <img src="https://github.com/user-attachments/assets/36636b6d-4a13-4bd5-980d-299169fb36c9" width="50%" alt="body_joints">
</div>

The following are related concept descriptions:

| **Concept**             | **Description**                                                                                                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Coordinate System       | Same world coordinate system as the headset data.                                                                                                                                                                 |
| Root Joint Node         | 0 (Pelvis)                                                                                                                                                                                                        |
| Parent/Child Joint Node | Nodes 1 to 23, the one closer to the root joint is the parent, the one closer to the limb end is the child.                                                                                                       |
| Bone                    | A rigid body between two nodes, its pose is stored in the parent node structure closer to the root joint. For example: the pose angle of the lower leg bone is stored in the Knee joint structure. More examples: |

The following is the BodyTrackerRole enumeration, corresponding one-to-one with the joints in the reference diagram.

```c#
public enum BodyTrackerRole
    {
        Pelvis = 0,  // Pelvis
        LEFT_HIP = 1,  // Left hip
        RIGHT_HIP = 2,  // Right hip
        SPINE1 = 3,  // Spine
        LEFT_KNEE = 4,  // Left knee
        RIGHT_KNEE = 5,  // Right knee
        SPINE2 = 6,  // Spine
        LEFT_ANKLE = 7,  // Left ankle
        RIGHT_ANKLE = 8,  // Right ankle
        SPINE3 = 9,  // Spine
        LEFT_FOOT = 10,  // Left foot
        RIGHT_FOOT = 11,  // Right foot
        NECK = 12,  // Neck
        LEFT_COLLAR = 13,  // Left collarbone
        RIGHT_COLLAR = 14,  // Right collarbone
        HEAD = 15,  // Head
        LEFT_SHOULDER = 16,  // Left shoulder
        RIGHT_SHOULDER = 17,  // Right shoulder
        LEFT_ELBOW = 18,  // Left elbow
        RIGHT_ELBOW = 19,  // Right elbow
        LEFT_WRIST = 20,  // Left wrist
        RIGHT_WRIST = 21,  // Right wrist
        LEFT_HAND = 22,  // Left hand
        RIGHT_HAND = 23  // Right hand
    }
```

JSON data description

```bash
{"dt":0,"flags":0,"timeStampNs":1732613438765715200,
    "joints":[{
    "p":"0.0,0.0,0.0,0.0,0.0,0.0,0.0",
    "t":0,
    "va":"0,0,0,0,0,0",
    "wva":"0,0,0,0,0,0"},.....}]}

```

| **key**     | **Type** | **Description**                                            |
| ----------- | -------- | ---------------------------------------------------------- |
| joints      | Json     | Array, represents 24 bones                                 |
| timeStampNs | int64    | Unix timestamp (nanoseconds)                               |
| p           | string   | Current bone's Pose (position and rotation, seven values)  |
| t           | long     | *IMU timestamp.*                                           |
| va          | string   | Position velocity (x,y,z) angular velocity (x,y,z)         |
| wva         | string   | Position acceleration (x,y,z) angular acceleration (x,y,z) |

#### 5. Tracker Independent Tracking

Original data of tracking Tracker

JSON data description

```bash
{"joints":[{"p":"0.0,-0.0,-0.0,0.0,0.0,-0.0,-0.0","va":"0.0,0.0,-0.0,0.0,0.0,0.0","wva":"0.0,0.0,-0.0,-0.0,0,0"},{"p":"-0.0,0.0,-0.0,-0.0,0.0,0.0,-0.0","va":"-0.0,0.0,0.0,0.0,-0.0,0.0","wva":"0.0,-0.0,-1.0,-0.0,-0.0,-0"}],"len":2,"timeStampNs":1733287634681455104,"sn":PC2310MLJ6050513G}
```

| **key**     | **Type** | **Description**                                                                                   |
| ----------- | -------- | ------------------------------------------------------------------------------------------------- |
| joints      | Json     | Array, represents all Tracker data (currently supports up to 3)                                   |
| timeStampNs | int64    | Unix timestamp (nanoseconds)                                                                      |
| p           | string   | Current bone's Pose (position and rotation, seven values)                                         |
| va          | string   | Position velocity (x,y,z) *Unit: millimeter*       Angular velocity (x,y,z) *Unit: meter*         |
| wva         | string   | Position acceleration (x,y,z) *Unit: millimeter*       Angular acceleration (x,y,z) *Unit: meter* |
| sn          | string   | Tracker serial number, used to distinguish different trackers                                     |


## Using the Python Bindings

**1. Get Controller and Headset Poses**

```python
import xrobotoolkit_sdk as xrt

xrt.init()

left_pose = xrt.get_left_controller_pose()
right_pose = xrt.get_right_controller_pose()
headset_pose = xrt.get_headset_pose()

print(f"Left Controller Pose: {left_pose}")
print(f"Right Controller Pose: {right_pose}")
print(f"Headset Pose: {headset_pose}")

xrt.close()
```

**2. Get Controller Inputs (Triggers, Grips, Buttons, Axes)**

```python
import xrobotoolkit_sdk as xrt

xrt.init()

# Triggers and Grips
left_trigger = xrt.get_left_trigger()
right_grip = xrt.get_right_grip()
print(f"Left Trigger: {left_trigger}, Right Grip: {right_grip}")

# Buttons
a_button_pressed = xrt.get_A_button()
x_button_pressed = xrt.get_X_button()
print(f"A Button Pressed: {a_button_pressed}, X Button Pressed: {x_button_pressed}")

# Axes
left_axis = xrt.get_left_axis()
right_axis_click = xrt.get_right_axis_click()
print(f"Left Axis: {left_axis}, Right Axis Clicked: {right_axis_click}")

# Timestamp
timestamp = xrt.get_time_stamp_ns()
print(f"Current Timestamp (ns): {timestamp}")

xrt.close()
```

**3. Get hand tracking state**
```python
import xrobotoolkit_sdk as xrt

xrt.init()

# Left Hand State
left_hand_tracking_state = xrt.get_left_hand_tracking_state()
print(f"Left Hand State: {left_hand_tracking_state}")

# Left Hand isActive
left_hand_is_active = xrt.get_left_hand_is_active()
print(f"Left Hand isActive: {left_hand_is_active}")

# Right Hand State
right_hand_tracking_state = xrt.get_right_hand_tracking_state()
print(f"Right Hand State: {right_hand_tracking_state}")

# Right Hand isActive
right_hand_is_active = xrt.get_right_hand_is_active()
print(f"Right Hand isActive: {right_hand_is_active}")

xrt.close()
```

**4. Get whole body motion tracking**
```python
import xrobotoolkit_sdk as xrt

xrt.init()

# Check if body tracking data is available
if xrt.is_body_data_available():
    # Get body joint poses (24 joints, 7 values each: x,y,z,qx,qy,qz,qw)
    body_poses = xrt.get_body_joints_pose()
    print(f"Body joints pose data: {body_poses}")
    
    # Get body joint velocities (24 joints, 6 values each: vx,vy,vz,wx,wy,wz)
    body_velocities = xrt.get_body_joints_velocity()
    print(f"Body joints velocity data: {body_velocities}")
    
    # Get body joint accelerations (24 joints, 6 values each: ax,ay,az,wax,way,waz)
    body_accelerations = xrt.get_body_joints_acceleration()
    print(f"Body joints acceleration data: {body_accelerations}")
    
    # Get IMU timestamps for each joint
    imu_timestamps = xrt.get_body_joints_timestamp()
    print(f"IMU timestamps: {imu_timestamps}")
    
    # Get body data timestamp
    body_timestamp = xrt.get_body_timestamp_ns()
    print(f"Body data timestamp: {body_timestamp}")
    
    # Example: Get specific joint data (Head joint is index 15)
    head_pose = body_poses[15]  # Head joint
    x, y, z, qx, qy, qz, qw = head_pose
    print(f"Head pose: Position({x:.3f}, {y:.3f}, {z:.3f}) Rotation({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
else:
    print("Body tracking data not available. Make sure:")
    print("1. PICO headset is connected")
    print("2. Body tracking is enabled in the control panel")
    print("3. At least two Pico Swift devices are connected and calibrated")

xrt.close()
```

**Body Joint Indices (24 joints total):**
- 0: Pelvis, 1: Left Hip, 2: Right Hip, 3: Spine1, 4: Left Knee, 5: Right Knee
- 6: Spine2, 7: Left Ankle, 8: Right Ankle, 9: Spine3, 10: Left Foot, 11: Right Foot
- 12: Neck, 13: Left Collar, 14: Right Collar, 15: Head, 16: Left Shoulder, 17: Right Shoulder
- 18: Left Elbow, 19: Right Elbow, 20: Left Wrist, 21: Right Wrist, 22: Left Hand, 23: Right Hand
