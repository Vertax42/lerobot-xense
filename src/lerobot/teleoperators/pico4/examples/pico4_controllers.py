import sys
import time

import xensevr_pc_service_sdk as xrt


def clear_screen():
    """Clear terminal and move cursor to top"""
    print("\033[2J\033[H", end="")


def run_tests():
    print("Starting Python binding test...")

    try:
        print("Initializing SDK...")
        xrt.init()
        print("SDK Initialized successfully.")
        time.sleep(1)

        i = 0
        last_zero_check_time = time.monotonic()
        zero_check_interval = 5.0  # Check every 5 seconds
        
        while True:
            clear_screen()

            print("=" * 60)
            print(f"  XenseVR Controller Data  |  Iteration: {i+1}")
            print("=" * 60)

            # Controller Poses
            left_pose = xrt.get_left_controller_pose()
            right_pose = xrt.get_right_controller_pose()
            
            # Headset Pose
            headset_pose = xrt.get_headset_pose()
            
            # Triggers & Grips
            left_trigger = xrt.get_left_trigger()
            right_trigger = xrt.get_right_trigger()
            left_grip = xrt.get_left_grip()
            right_grip = xrt.get_right_grip()
            
            # Check if all data is zero (indicates Pico VR client is not running)
            # Only check every 5 seconds to reduce computation overhead
            current_time = time.monotonic()
            if current_time - last_zero_check_time >= zero_check_interval:
                all_zero = (
                    all(abs(v) < 1e-6 for v in left_pose) and
                    all(abs(v) < 1e-6 for v in right_pose) and
                    all(abs(v) < 1e-6 for v in headset_pose) and
                    abs(left_trigger) < 1e-6 and
                    abs(right_trigger) < 1e-6 and
                    abs(left_grip) < 1e-6 and
                    abs(right_grip) < 1e-6
                )
                
                if all_zero:
                    print("\n" + "=" * 60)
                    print("  ⚠️  WARNING: All data is zero!")
                    print("  Pico VR client may not be running.")
                    print("  Please ensure the Pico VR service is started.")
                    print("=" * 60 + "\n")
                
                last_zero_check_time = current_time
            
            print(f"\n[Left Controller Pose]")
            print(f"  Position:    x={left_pose[0]:8.4f}  y={left_pose[1]:8.4f}  z={left_pose[2]:8.4f}")
            print(f"  Quaternion: qx={left_pose[3]:8.4f} qy={left_pose[4]:8.4f} qz={left_pose[5]:8.4f} qw={left_pose[6]:8.4f}")

            print(f"\n[Right Controller Pose]")
            print(f"  Position:    x={right_pose[0]:8.4f}  y={right_pose[1]:8.4f}  z={right_pose[2]:8.4f}")
            print(f"  Quaternion: qx={right_pose[3]:8.4f} qy={right_pose[4]:8.4f} qz={right_pose[5]:8.4f} qw={right_pose[6]:8.4f}")

            print(f"\n[Headset Pose]")
            print(f"  Position:    x={headset_pose[0]:8.4f}  y={headset_pose[1]:8.4f}  z={headset_pose[2]:8.4f}")
            print(f"  Quaternion: qx={headset_pose[3]:8.4f} qy={headset_pose[4]:8.4f} qz={headset_pose[5]:8.4f} qw={headset_pose[6]:8.4f}")

            print(f"\n[Inputs]")
            print(f"  Left  Trigger: {left_trigger:6.3f}    Grip: {left_grip:6.3f}")
            print(f"  Right Trigger: {right_trigger:6.3f}    Grip: {right_grip:6.3f}")

            # Motion Trackers
            num_trackers = xrt.num_motion_data_available()
            if num_trackers > 0:
                tracker_poses = xrt.get_motion_tracker_pose()
                tracker_velocities = xrt.get_motion_tracker_velocity()
                tracker_accelerations = xrt.get_motion_tracker_acceleration()
                tracker_serial_numbers = xrt.get_motion_tracker_serial_numbers()

                print(f"\n[Motion Trackers]  ({num_trackers} tracker(s) available)")
                for idx in range(num_trackers):
                    pose = tracker_poses[idx]
                    vel = tracker_velocities[idx]
                    acc = tracker_accelerations[idx]
                    sn = tracker_serial_numbers[idx] if idx < len(tracker_serial_numbers) else "N/A"

                    print(f"\n  Tracker {idx + 1} (SN: {sn})")
                    print(f"    Position:     x={pose[0]:8.4f}  y={pose[1]:8.4f}  z={pose[2]:8.4f}")
                    print(f"    Quaternion:  qx={pose[3]:8.4f} qy={pose[4]:8.4f} qz={pose[5]:8.4f} qw={pose[6]:8.4f}")
                    print(f"    Linear Vel:  vx={vel[0]:8.4f} vy={vel[1]:8.4f} vz={vel[2]:8.4f}")
                    print(f"    Angular Vel: wx={vel[3]:8.4f} wy={vel[4]:8.4f} wz={vel[5]:8.4f}")
                    print(f"    Linear Acc:  ax={acc[0]:8.4f} ay={acc[1]:8.4f} az={acc[2]:8.4f}")
                    print(f"    Angular Acc: wax={acc[3]:8.4f} way={acc[4]:8.4f} waz={acc[5]:8.4f}")
            else:
                print(f"\n[Motion Trackers]  No trackers available")

            print("\n" + "=" * 60)
            print("  Press Ctrl+C to exit")
            print("=" * 60)

            sys.stdout.flush()
            time.sleep(0.02)
            i += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except RuntimeError as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        print("\nClosing SDK...")
        xrt.close()
        print("SDK closed.")
        print("Test finished.")


if __name__ == "__main__":
    run_tests()
