import sys

# Work around conda environments that ship `$CONDA_PREFIX/lib/udev/` (a directory),
# which can break `pyudev` and therefore `xensesdk` device scanning.
from lerobot.cameras.xense.camera_xense import _patch_ctypes_find_library_for_udev
_patch_ctypes_find_library_for_udev()

# Now safe to import xensesdk
from xensesdk import ExampleView
from xensesdk import Sensor, CameraSource


def main():

    sensor_0 = Sensor.create("OG000337", use_gpu=True   , api=CameraSource.CV2_V4L2)
    View = ExampleView(sensor_0)
    View2d = View.create2d(
        Sensor.OutputType.Difference,
        Sensor.OutputType.Depth,
        Sensor.OutputType.Marker2D,
    )

    def callback():
        force, res_force, mesh_init, src, diff, depth = sensor_0.selectSensorInfo(
            Sensor.OutputType.Force,
            Sensor.OutputType.ForceResultant,
            Sensor.OutputType.Mesh3DInit,
            Sensor.OutputType.Rectify,
            Sensor.OutputType.Difference,
            Sensor.OutputType.Depth,
        )
        marker_img = sensor_0.drawMarkerMove(src)
        View2d.setData(Sensor.OutputType.Marker2D, marker_img)
        View2d.setData(Sensor.OutputType.Difference, diff)
        View2d.setData(Sensor.OutputType.Depth, depth)
        View.setForceFlow(force, res_force, mesh_init)
        View.setDepth(depth)

    View.setCallback(callback)
    View.show()
    sensor_0.release()
    sys.exit()


if __name__ == "__main__":
    main()
