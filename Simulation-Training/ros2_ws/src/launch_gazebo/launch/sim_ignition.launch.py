# Copyright 2023 Clearpath Robotics, Inc.
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
#
# @author Roni Kreinin (rkreinin@clearpathrobotics.com)

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


import sys

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='false',
                          choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('world', default_value='warehouse',
                          description='Ignition World'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
    DeclareLaunchArgument('num', default_value='1',
                            description='Number of robots to spawn'),
]

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(pose_element, default_value='0.0',
                     description=f'{pose_element} component of the robot pose.'))


def generate_launch_description():

    num = 1
    for arg in sys.argv:
        if arg.startswith('num'):
            num = int(arg.split(':=')[1])

            ''' for i in range(num):
                ARGUMENTS.append(DeclareLaunchArgument(f'pose_x_{i}', default_value='0.0',
                     description=f'x component of the robot pose.'))

                ARGUMENTS.append(DeclareLaunchArgument(f'pose_y_{i}', default_value='0.0',
                     description=f'y component of the robot pose.'))

                ARGUMENTS.append(DeclareLaunchArgument(f'pose_z_{i}', default_value='0.0',
                     description=f'z component of the robot pose.'))

                ARGUMENTS.append(DeclareLaunchArgument(f'pose_yaw_{i}', default_value='0.0',
                     description=f'yaw component of the robot pose.'))
            '''


    # Directories
    pkg_turtlebot4_ignition_bringup = get_package_share_directory(
        'turtlebot4_ignition_bringup')

    pkg_launch_gazebo = get_package_share_directory('launch_gazebo')

    # Paths
    ignition_launch = PathJoinSubstitution(
        [pkg_turtlebot4_ignition_bringup, 'launch', 'ignition.launch.py'])
    robot_spawn_launch = PathJoinSubstitution(
        [pkg_turtlebot4_ignition_bringup, 'launch', 'turtlebot4_spawn.launch.py'])

    ignition = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ignition_launch]),
        launch_arguments=[
            ('world', LaunchConfiguration('world'))
        ]
    )
    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(ignition)


    print("num: ", num)

    for i in range(num):
        print("i: ", i)
        robot_spawn = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([robot_spawn_launch]),
            launch_arguments=[
                ('namespace', 'turtlebot4_' + str(i)),
                ('rviz', LaunchConfiguration('rviz')),
                ('x', str(1 * i)),
                ('y', str(1)),
                ('z', str(0)),
                ('yaw',str(0))]
        )
        ld.add_action(robot_spawn)

    
    return ld
