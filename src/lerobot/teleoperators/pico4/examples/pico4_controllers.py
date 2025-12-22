import sys
import time
import statistics

import xensevr_pc_service_sdk as xrt


def clear_screen():
    """Clear terminal and move cursor to top"""
    print("\033[2J\033[H", end="")


def calculate_position_delta(pose1, pose2):
    """Calculate position delta between two poses"""
    if pose1 is None or pose2 is None:
        return 0.0
    dx = pose2[0] - pose1[0]
    dy = pose2[1] - pose1[1]
    dz = pose2[2] - pose1[2]
    return (dx**2 + dy**2 + dz**2) ** 0.5


def run_tests():
    print("Starting Python binding test with stability diagnostics...")

    try:
        print("Initializing SDK...")
        xrt.init()
        print("SDK Initialized successfully.")
        time.sleep(1)

        i = 0
        last_zero_check_time = time.monotonic()
        zero_check_interval = 5.0  # Check every 5 seconds
        
        # Stability tracking
        read_times = []  # SDK read latency
        position_deltas = []  # Position change between frames
        last_right_pose = None
        max_read_time = 0.0
        max_position_delta = 0.0
        jump_count = 0  # Count of position jumps > threshold
        jump_threshold = 0.01  # 1cm jump threshold
        
        # For running statistics (last N samples)
        max_samples = 500
        
        while True:
            loop_start = time.perf_counter()
            
            # ===== Measure SDK read latency =====
            read_start = time.perf_counter()
            
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
            
            # A/X buttons
            a_button = xrt.get_A_button()
            x_button = xrt.get_X_button()
            
            read_end = time.perf_counter()
            read_time_ms = (read_end - read_start) * 1000
            
            # Track read latency
            read_times.append(read_time_ms)
            if len(read_times) > max_samples:
                read_times.pop(0)
            if read_time_ms > max_read_time:
                max_read_time = read_time_ms
            
            # ===== Track position continuity =====
            if last_right_pose is not None:
                delta = calculate_position_delta(last_right_pose, right_pose)
                position_deltas.append(delta)
                if len(position_deltas) > max_samples:
                    position_deltas.pop(0)
                if delta > max_position_delta:
                    max_position_delta = delta
                if delta > jump_threshold:
                    jump_count += 1
            last_right_pose = right_pose
            
            # Only update display every 10 iterations to reduce overhead
            if i % 10 == 0:
                clear_screen()

                print("=" * 70)
                print(f"  XenseVR Controller Data  |  Iteration: {i+1}")
                print("=" * 70)

                # ===== Stability Statistics =====
                print("\n[SDK Stability Diagnostics]")
                if read_times:
                    avg_read = statistics.mean(read_times)
                    print(f"  SDK Read Latency:  avg={avg_read:.2f}ms  max={max_read_time:.2f}ms  current={read_time_ms:.2f}ms")
                if position_deltas:
                    avg_delta = statistics.mean(position_deltas) * 1000  # Convert to mm
                    max_delta_mm = max_position_delta * 1000
                    print(f"  Position Delta:    avg={avg_delta:.2f}mm  max={max_delta_mm:.2f}mm")
                    print(f"  Position Jumps:    {jump_count} (>{jump_threshold*1000:.0f}mm)")
                
                # Stability assessment
                if read_times and len(read_times) > 10:
                    if max_read_time > 50:
                        print(f"  ⚠️  WARNING: High SDK latency detected ({max_read_time:.1f}ms)")
                    else:
                        print(f"  ✓  SDK latency OK")
                if jump_count > 0:
                    jump_rate = jump_count / (i + 1) * 100
                    if jump_rate > 1:
                        print(f"  ⚠️  WARNING: Position jumps detected ({jump_rate:.1f}% of frames)")
                    else:
                        print(f"  ✓  Position continuity OK (jump rate: {jump_rate:.2f}%)")
                
                # Check if all data is zero
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
                        print("\n" + "=" * 70)
                        print("  ⚠️  WARNING: All data is zero!")
                        print("  Pico VR client may not be running.")
                        print("=" * 70 + "\n")
                    
                    last_zero_check_time = current_time
                
                print(f"\n[Right Controller Pose]")
                print(f"  Position:    x={right_pose[0]:8.4f}  y={right_pose[1]:8.4f}  z={right_pose[2]:8.4f}")
                print(f"  Quaternion: qx={right_pose[3]:8.4f} qy={right_pose[4]:8.4f} qz={right_pose[5]:8.4f} qw={right_pose[6]:8.4f}")

                print(f"\n[Left Controller Pose]")
                print(f"  Position:    x={left_pose[0]:8.4f}  y={left_pose[1]:8.4f}  z={left_pose[2]:8.4f}")
                print(f"  Quaternion: qx={left_pose[3]:8.4f} qy={left_pose[4]:8.4f} qz={left_pose[5]:8.4f} qw={left_pose[6]:8.4f}")

                print(f"\n[Inputs]")
                print(f"  Left  Trigger: {left_trigger:6.3f}    Grip: {left_grip:6.3f}    X Button: {x_button}")
                print(f"  Right Trigger: {right_trigger:6.3f}    Grip: {right_grip:6.3f}    A Button: {a_button}")

                # Motion Trackers
                num_trackers = xrt.num_motion_data_available()
                if num_trackers > 0:
                    tracker_poses = xrt.get_motion_tracker_pose()
                    tracker_serial_numbers = xrt.get_motion_tracker_serial_numbers()

                    print(f"\n[Motion Trackers]  ({num_trackers} tracker(s) available)")
                    for idx in range(min(num_trackers, 2)):  # Only show first 2 to save space
                        pose = tracker_poses[idx]
                        sn = tracker_serial_numbers[idx] if idx < len(tracker_serial_numbers) else "N/A"
                        print(f"  Tracker {idx + 1} (SN: {sn}): x={pose[0]:6.3f} y={pose[1]:6.3f} z={pose[2]:6.3f}")
                else:
                    print(f"\n[Motion Trackers]  No trackers available")

                # Loop timing
                loop_time_ms = (time.perf_counter() - loop_start) * 1000
                target_hz = 100
                print(f"\n[Loop Timing]")
                print(f"  Loop time: {loop_time_ms:.2f}ms ({1000/max(loop_time_ms, 0.1):.0f} Hz)")

                print("\n" + "=" * 70)
                print("  Press Ctrl+C to exit and see summary")
                print("=" * 70)

                sys.stdout.flush()
            
            # Target ~100Hz (10ms per loop)
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, 0.01 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            i += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("\n" + "=" * 70)
        print("  FINAL STABILITY REPORT")
        print("=" * 70)
        if read_times:
            print(f"  Total iterations: {i}")
            print(f"  SDK Read Latency:")
            print(f"    Average: {statistics.mean(read_times):.2f}ms")
            print(f"    Max:     {max_read_time:.2f}ms")
            print(f"    Std Dev: {statistics.stdev(read_times) if len(read_times) > 1 else 0:.2f}ms")
        if position_deltas:
            print(f"  Position Continuity:")
            print(f"    Avg Delta: {statistics.mean(position_deltas)*1000:.2f}mm")
            print(f"    Max Delta: {max_position_delta*1000:.2f}mm")
            print(f"    Jump Count: {jump_count} (threshold: {jump_threshold*1000:.0f}mm)")
            print(f"    Jump Rate:  {jump_count/i*100:.2f}%")
        print("=" * 70)
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
