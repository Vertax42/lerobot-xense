import sys
from xensesdk import ExampleView
from xensesdk import Sensor
from xensesdk import call_service


def main():
    MASTER_SERVICE = "master_e2b26adbb104"
    # find all sensors
    ret = call_service(MASTER_SERVICE, "scan_sensor_sn")
    if ret is None:
        print(f"Failed to scan sensors")
        sys.exit(1)
    else:
        print(f"Found sensors: {ret}, using the first one.")
    serial_number = list(ret.keys())[1]

    # create a sensor
    sensor_0 = Sensor.create(serial_number, mac_addr=MASTER_SERVICE.split("_")[-1], rectify_size=(200, 350))
    View = ExampleView(sensor_0)
    View2d = View.create2d(Sensor.OutputType.Difference, Sensor.OutputType.Rectify)
    
    def callback():
        diff, rectify = sensor_0.selectSensorInfo(Sensor.OutputType.Difference, Sensor.OutputType.Rectify)
        View2d.setData(Sensor.OutputType.Difference, diff)
        View2d.setData(Sensor.OutputType.Rectify, rectify)
        # View.setDepth(depth)
    View.setCallback(callback)

    View.show()
    sensor_0.release()
    sys.exit()


if __name__ == '__main__':
    main()