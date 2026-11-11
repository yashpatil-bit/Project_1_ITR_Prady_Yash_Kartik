"""Keyboard emergency stop - run it in its own terminal during demos.

    ros2 run peg_sorting estop
SPACE (or any key except r) -> /emergency_stop true (driver aborts motion and holds position)
r                            -> /emergency_stop false (release)
"""
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from ..utils.keyboard import KeyReader


def main(args=None):
    rclpy.init(args=args)
    node = Node('estop_keyboard')
    pub = node.create_publisher(Bool, 'emergency_stop', 10)
    print('EMERGENCY STOP: press SPACE to stop the robot, r to release, Ctrl-C to quit this tool.')
    try:
        with KeyReader() as kr:
            if not kr.available:
                print('needs a terminal')
                return
            while rclpy.ok():
                k = kr.get(0.1)
                if k is None:
                    continue
                if k == 'r':
                    pub.publish(Bool(data=False))
                    print('released')
                else:
                    for _ in range(3):
                        pub.publish(Bool(data=True))
                        time.sleep(0.01)
                    print('!!! EMERGENCY STOP SENT !!!  (r = release)')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
