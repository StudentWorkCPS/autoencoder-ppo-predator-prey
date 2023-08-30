# Launches The robots in Gazebo and RViz
#
import os  
import rclpy
import sys

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
import launch_ros



def generate_launch_description(): 
    include_predator_num = LaunchConfiguration('predator_num')
    inlude_prey_num = LaunchConfiguration('prey_num')
    include_env_num = LaunchConfiguration('env_num')
    include_server = LaunchConfiguration('server')
    include_gui = LaunchConfiguration('gui')
    include_model = LaunchConfiguration('model', default='thymio')
    
    env_num = 1
    model = 'thymio'
    for arg in sys.argv:
        if arg.startswith('predator_num'):
            predator_num = int(arg.split(':=')[1])
        if arg.startswith('prey_num'):
            prey_num = int(arg.split(':=')[1])
        if arg.startswith('env_num'):
            env_num = int(arg.split(':=')[1])
        if arg.startswith('model'):
            model = arg.split(':=')[1]
 
    
        

    print(f'predator_num: {predator_num}')
    print(f'prey_num: {prey_num}')
    print(f'env_num: {env_num}')
    print(f'model: {model}')

    robots = LaunchDescription()
    # in row
    row = lambda i,num: (i - num/2 + 1/2)*0.4


    for eid in range(env_num):
        env_x = eid*4.2
        env_y = 0.0

        if eid != 0:
            env_node = launch_ros.actions.Node(package='gazebo_ros', executable='spawn_entity.py', output='screen', 
                                                arguments=['-file', os.path.join(get_package_share_directory('launch_gazebo'),'objects','4x4.sdf'), 
                                                            '-entity', 'env'+str(eid), 
                                                            '-x', str(env_x), '-y', str(env_y), '-z', '0.0'])
            robots.add_action(env_node)

        for i in range(0,predator_num):
            robot_x = env_x - 0.95
            robot_y = env_y + row(i,predator_num)


            robot_node = launch_ros.actions.Node(package='launch_gazebo', executable='spawn_robot', output='screen', 
                                                arguments=['-file', model + '.sdf', 
                                                            '-name', 'env{}_predator{}'.format(eid,i),
                                                            '-x',str(robot_x), '-y', str(robot_y), '-z', '0.0', 
                                                            '-type','predator'])
            robots.add_action(robot_node)

        for i in range(0,prey_num):
            robot_x = env_x + 1.0
            robot_y = env_y + row(i,prey_num)

            robot_node = launch_ros.actions.Node(package='launch_gazebo', executable='spawn_robot', output='screen', 
                                                arguments=['-file', model + '.sdf', 
                                                            '-name', 'env{}_prey{}'.format(eid,i), 
                                                            '-x', str(robot_x) , '-y', str(robot_y), '-z', '0.0', 
                                                            '-type','prey'])
            robots.add_action(robot_node)

    return LaunchDescription([
            DeclareLaunchArgument('gui', default_value='true',
                                description='Set to "false" to run headless.'),

            DeclareLaunchArgument('server', default_value='true',
                                description='Set to "false" not to run gzserver.'),
            DeclareLaunchArgument('predator_num',default_value='1',
                                description='Set number of preys to spawn.'),
            DeclareLaunchArgument('prey_num',default_value='1',description='Set number of preys to spawn'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([get_package_share_directory('gazebo_ros'),'/launch', '/gzserver.launch.py']),
                launch_arguments={'verbose':'true',
                                  'world': os.path.join(get_package_share_directory('launch_gazebo'),'worlds', '4x4.world'),
                                  
                                  }.items(),
                condition=IfCondition(include_server)
            ),
            
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([get_package_share_directory('gazebo_ros'),'/launch', '/gzclient.launch.py']),
                condition=IfCondition(include_gui)
            ),

            robots
        ])

def main(args=None):
    rclpy.init()

    ld = generate_launch_description()

    ls = LaunchService()
    # set arguments to launch description   
    ls.include_launch_description(ld)
    
    ls.run()
    rclpy.shutdown()