"""Observe SDK polling, pose changes and stream timestamps without filtering input."""

from collections.abc import Sequence


class InputTiming:
    """Local repeat durations, not one-way latency or controller tracking status.

    The SDK timestamp describes the whole device stream, not one controller.
    An unchanged pose can also mean the user is holding still. Compare the
    counters while deliberately moving the controller to diagnose freezes.
    """

    def __init__(self, report_interval: float = 0.5):
        self.report_interval = report_interval
        self._report_at: float | None = None
        self._poll_at: float | None = None
        self._pose: tuple[float, ...] | None = None
        self._stamp: int | None = None
        self._pose_changed_at = 0.0
        self._stamp_changed_at = 0.0
        self._polls = self._pose_changes = self._stamp_changes = 0
        self._max_poll_gap = self._max_pose_repeat = self._max_stamp_repeat = 0.0

    def observe(self, now: float, pose: Sequence[float], stamp: int | None) -> dict[str, float] | None:
        pose = tuple(pose)  # Compare original SDK precision, before float32 conversion.
        stamp = stamp if stamp is not None and stamp > 0 else None
        if self._report_at is None:
            self._report_at = self._poll_at = now
            self._pose_changed_at = self._stamp_changed_at = now
            self._pose, self._stamp = pose, stamp
            return None

        self._polls += 1
        self._max_poll_gap = max(self._max_poll_gap, now - self._poll_at)
        self._poll_at = now
        # Include the gap ending at this poll, even when fresh data just arrived.
        self._max_pose_repeat = max(self._max_pose_repeat, now - self._pose_changed_at)
        if pose != self._pose:
            self._pose_changes += 1
            self._pose_changed_at = now
            self._pose = pose
        if stamp is not None and self._stamp is not None:
            self._max_stamp_repeat = max(self._max_stamp_repeat, now - self._stamp_changed_at)
        if stamp != self._stamp:
            self._stamp_changes += int(stamp is not None)
            self._stamp_changed_at = now
            self._stamp = stamp

        elapsed = now - self._report_at
        if elapsed < self.report_interval:
            return None
        result = {
            "poll_hz": self._polls / elapsed,
            "pose_changes": self._pose_changes,
            "sdk_stamp_changes": self._stamp_changes,
            "pose_repeat_ms": (now - self._pose_changed_at) * 1000,
            "sdk_stamp_repeat_ms": (now - self._stamp_changed_at) * 1000 if stamp else -1.0,
            "max_poll_gap_ms": self._max_poll_gap * 1000,
            "max_pose_repeat_ms": self._max_pose_repeat * 1000,
            "max_sdk_stamp_repeat_ms": self._max_stamp_repeat * 1000 if stamp else -1.0,
        }
        self._report_at = now
        self._polls = self._pose_changes = self._stamp_changes = 0
        self._max_poll_gap = self._max_pose_repeat = self._max_stamp_repeat = 0.0
        return result

    @staticmethod
    def format_report(report: dict[str, float]) -> str:
        return " ".join(f"{key}={value:.1f}" for key, value in report.items())
