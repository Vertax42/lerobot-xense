#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
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

"""Group USB devices by the hub they sit behind.

Both gripper families wire one arm's devices — the gripper's own serial device,
its wrist camera, and its two visuotactile sensors — behind a single per-side USB
hub. That shared hub is what lets a driver work out which camera belongs to which
arm without any serial number written into a config: resolve the side from the
gripper (authoritative), read its hub, and claim everything else on that hub.

This module holds the robot-agnostic half of that: sysfs walking, hub/port token
extraction, and the two enumerations (visuotactile sensors via xensesdk, video
devices via v4l2-ctl) keyed by hub. Which device anchors a side is the caller's
business — see ``taccap_discovery`` (firmware SN) and ``serial_discovery``
(board-SN parity).

Everything here is read-only inspection of /sys and /dev; nothing opens a device.
"""

import os
import re

from lerobot.utils.robot_utils import get_logger

logger = get_logger("UsbTopology")

# sysfs path like ".../usb1/1-3/1-3.1/1-3.1:1.2/tty/ttyACM1" — the top hub token
# ("1-3") groups every device physically behind one hub, and the full port token
# ("1-3.1") orders devices within that hub.
_HUB_RE = re.compile(r"/usb\d+/(\d+-\d+)/")
_PORT_RE = re.compile(r"/(\d+-\d+(?:\.\d+)+)/")


def sysfs_device_dir(dev_class: str, node: str) -> str:
    """Realpath of ``/sys/class/<dev_class>/<node>/device`` (follows symlinks)."""
    return os.path.realpath(f"/sys/class/{dev_class}/{os.path.basename(node)}/device")


def usb_hub_and_port(sysfs_dir: str) -> tuple[str | None, str | None]:
    """Extract the USB hub token (e.g. '1-3') and full port token (e.g. '1-3.2')."""
    padded = sysfs_dir + "/"
    hub = _HUB_RE.search(padded)
    port = _PORT_RE.search(padded)
    return (hub.group(1) if hub else None, port.group(1) if port else None)


def video_node(cam) -> str:
    """xensesdk ``scanSerialNumber()`` maps SN → cam id (int) or a /dev path."""
    if isinstance(cam, str) and cam.startswith("/dev/"):
        return cam
    return f"/dev/video{int(cam)}"


def usb_vid_pid(device: str) -> tuple[str, str] | None:
    """USB ``(vendor_id, product_id)`` behind a tty node, lowercase hex, or None.

    Walks up from the tty's sysfs node to the first ancestor carrying idVendor —
    that is the USB device, whereas the tty itself sits on an interface.
    """
    path = sysfs_device_dir("tty", device)
    while path and path != "/":
        vid = os.path.join(path, "idVendor")
        pid = os.path.join(path, "idProduct")
        if os.path.isfile(vid) and os.path.isfile(pid):
            try:
                with open(vid) as f_v, open(pid) as f_p:
                    return f_v.read().strip().lower(), f_p.read().strip().lower()
            except OSError:
                return None
        path = os.path.dirname(path)
    return None


def hub_of_serial_device(device: str) -> str | None:
    """USB hub token behind which a serial device (tty) sits.

    ``device`` may be a /dev/serial/by-id symlink or a /dev/ttyUSB*, /dev/ttyACM*
    path; symlinks are resolved first.
    """
    tty = os.path.realpath(device)
    hub, _ = usb_hub_and_port(sysfs_device_dir("tty", tty))
    return hub


def hub_of_video_node(node: str) -> tuple[str | None, str | None]:
    """(hub, port) tokens for a /dev/videoN node."""
    return usb_hub_and_port(sysfs_device_dir("video4linux", node))


def tactile_sns_by_hub() -> dict[str, list[tuple[str, str]]]:
    """Enumerate visuotactile sensors and group them by USB hub.

    Returns:
        ``{hub: [(usb_port, serial), ...]}``, each list sorted by USB port so a
        side's sensors come back in a stable order run to run — which keeps logs
        and error messages comparable between runs. It is *not* what names the
        pads: the ``left`` / ``right`` in ``*_tactile_<finger>`` comes from each
        serial's own parity (``camera_injection.tactile_finger``), so a sensor
        moved to the other USB port keeps its key.

    Raises:
        RuntimeError: If xensesdk is unavailable or the scan fails.
    """
    try:
        from xensesdk import Sensor  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "xensesdk is not importable — needed to enumerate visuotactile "
            f"sensors. Install the xensesdk wheel. Original error: {e!r}"
        ) from e

    try:
        scanned = Sensor.scanSerialNumber()  # {serial: cam_id}
    except Exception as e:
        raise RuntimeError(f"xensesdk Sensor.scanSerialNumber() failed: {e}") from e

    by_hub: dict[str, list[tuple[str, str]]] = {}
    for sn, cam in scanned.items():
        node = video_node(cam)
        hub, port = hub_of_video_node(node)
        if hub is None:
            logger.warn(f"Could not resolve USB hub for tactile sensor {sn} ({node}); skipping.")
            continue
        by_hub.setdefault(hub, []).append((port or node, sn))
    for entries in by_hub.values():
        entries.sort()
    return by_hub


def video_names_by_hub() -> dict[str, list[tuple[str, str]]]:
    """Enumerate V4L2 devices by name and group them by USB hub.

    The names are exactly what ``v4l2-ctl --list-devices`` reports, which is the
    same namespace ``OpenCVCameraConfig.index_or_path`` resolves against (see
    ``lerobot.cameras.opencv.camera_opencv._resolve_v4l2_device_name``) — so a
    name from here can be handed straight to a camera config.

    Returns:
        ``{hub: [(usb_port, name), ...]}``, sorted by USB port. A device exposing
        several /dev/video* nodes is listed once, under its first node's hub.
    """
    from lerobot.cameras.opencv.camera_opencv import _parse_v4l2_devices

    by_hub: dict[str, list[tuple[str, str]]] = {}
    for name, paths in _parse_v4l2_devices().items():
        if not paths:
            continue
        hub, port = hub_of_video_node(paths[0])
        if hub is None:
            logger.warn(f"Could not resolve USB hub for video device {name} ({paths[0]}); skipping.")
            continue
        by_hub.setdefault(hub, []).append((port or paths[0], name))
    for entries in by_hub.values():
        entries.sort()
    return by_hub
