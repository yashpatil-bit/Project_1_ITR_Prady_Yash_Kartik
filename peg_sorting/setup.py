import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'peg_sorting'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Prady, Yash, Kartik',
    maintainer_email='team@example.com',
    description='Vision-guided peg sorting with the ROBOTIS Manipulator X',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'driver = peg_sorting.nodes.driver_node:main',
            'vision = peg_sorting.nodes.vision_node:main',
            'task_manager = peg_sorting.nodes.task_manager:main',
            'teleop_keyboard = peg_sorting.nodes.teleop_keyboard:main',
            'estop = peg_sorting.nodes.estop:main',
            'scan_dynamixel = peg_sorting.tools.scan_dynamixel:main',
            'hsv_tuner = peg_sorting.tools.hsv_tuner:main',
            'calibrate_aruco = peg_sorting.tools.calibrate_aruco:main',
            'calibrate_handeye = peg_sorting.tools.calibrate_handeye:main',
            'check_calibration = peg_sorting.tools.check_calibration:main',
            'capture_dataset = peg_sorting.tools.capture_dataset:main',
            'train_yolo = peg_sorting.tools.train_yolo:main',
            'evaluate_timing = peg_sorting.tools.evaluate_timing:main',
            'suggest_board_pose = peg_sorting.tools.suggest_board_pose:main',
            'calibrate_workspace = peg_sorting.tools.calibrate_workspace:main',
        ],
    },
)
