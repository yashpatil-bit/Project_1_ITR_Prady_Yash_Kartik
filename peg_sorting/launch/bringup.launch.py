"""Driver + vision.   ros2 launch peg_sorting bringup.launch.py [mock:=true] [show_window:=true]"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mock = LaunchConfiguration('mock')
    show = LaunchConfiguration('show_window')
    port = LaunchConfiguration('port')
    return LaunchDescription([
        DeclareLaunchArgument('mock', default_value='false', description='simulate robot + camera'),
        DeclareLaunchArgument('show_window', default_value='true', description='OpenCV window with detections'),
        DeclareLaunchArgument('port', default_value='', description='U2D2 serial port (default: robot.yaml)'),
        DeclareLaunchArgument('sim_scene', default_value='sim_scene.yaml', description='mock scene file in config/'),
        Node(package='peg_sorting', executable='driver', name='manipulator_driver', output='screen',
             parameters=[{'mock': mock, 'port': port}]),
        Node(package='peg_sorting', executable='vision', name='vision', output='screen',
             parameters=[{'mock': mock, 'show_window': show, 'sim_scene': LaunchConfiguration('sim_scene')}]),
    ])
