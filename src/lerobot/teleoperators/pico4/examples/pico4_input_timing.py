"""Read Pico input timing without importing or connecting any robot driver.

Run while moving the chosen controller continuously. Stream timestamp repeats
suggest stale SDK data; pose-only repeats can indicate upstream tracking freezes
or a stationary controller. Neither is a direct measurement of network latency.
"""

import argparse
import time

from lerobot.teleoperators.pico4.input_timing import InputTiming


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=100.0)
    args = parser.parse_args()
    if not 0 < args.seconds < float("inf") or not 0 < args.fps <= 1000:
        parser.error("seconds must be finite and positive; fps must be in (0, 1000]")

    import xensevr_pc_service_sdk as xrt

    get_pose = getattr(xrt, f"get_{args.side}_controller_pose")
    get_grip = getattr(xrt, f"get_{args.side}_grip")
    get_stamp = getattr(xrt, "get_time_stamp_ns", None)
    timing = InputTiming()
    print(f"Reading {args.side} Pico controller only; no robot connection. Move it continuously.")
    print("*_repeat_ms = local time without an observed change; timestamp -1 = unavailable.")
    xrt.init()
    try:
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            tick = time.monotonic()
            pose, grip = get_pose(), float(get_grip())
            stamp = int(get_stamp()) if get_stamp is not None else None
            report = timing.observe(time.monotonic(), pose, stamp)
            if report is not None:
                print(
                    f"[PICO_INPUT] side={args.side} {InputTiming.format_report(report)} "
                    f"grip={grip:.3f} raw_pose={[round(float(v), 7) for v in pose]}",
                    flush=True,
                )
            time.sleep(max(0.0, tick + 1 / args.fps - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        xrt.close()


if __name__ == "__main__":
    main()
