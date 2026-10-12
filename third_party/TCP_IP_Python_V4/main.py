# from lerobot.robots.dobot_nova5.TCP_IP_Python_V4.DobotDemo import DobotDemo
from DobotDemo import DobotDemo

if __name__ == '__main__':
    dobot = DobotDemo("192.168.5.102")
    dobot.start()
