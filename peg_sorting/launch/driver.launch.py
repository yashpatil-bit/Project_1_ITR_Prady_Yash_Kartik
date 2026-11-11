"""Driver only (teleop without vision).   ros2 launch peg_sorting driver.launch.py [mock:=true]"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mock', default_value='false'),
        DeclareLaunchArgument('port', default_value=''),
        Node(package='peg_sorting', executable='driver', name='manipulator_driver', output='screen',
             parameters=[{'mock': LaunchConfiguration('mock'), 'port': LaunchConfiguration('port')}]),
    ])
