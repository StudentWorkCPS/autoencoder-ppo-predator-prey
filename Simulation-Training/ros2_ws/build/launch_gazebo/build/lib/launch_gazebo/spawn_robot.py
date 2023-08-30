import os  
import rclpy
import argparse

from ament_index_python.packages import get_package_share_directory

from gazebo_msgs.srv import SpawnEntity

def main():

    parser = argparse.ArgumentParser(description='Spawn a robot in gazebo')
    parser.add_argument('-file', type=str, default='thymio.sdf', help='Path to the robot model')
    parser.add_argument('-name', type=str, help='Name of the robot and namespace')
    parser.add_argument('-x',type=float, help='X position of the robot')
    parser.add_argument('-y',type=float, help='Y position of the robot')
    parser.add_argument('-z',type=float, help='Z position of the robot')
    parser.add_argument('-type',type=str, help='Wether is it a prey or a predator')
    args , unknown = parser.parse_known_args()

    rclpy.init()
    node = rclpy.create_node('spawn_robot')
    node.get_logger().info('Waiting for service /spawn_entity...')
    client = node.create_client(SpawnEntity, '/spawn_entity')
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')


    request = SpawnEntity.Request()
    request.name = args.name
    pkg_path = get_package_share_directory('launch_gazebo')
    xml = open(os.path.join(pkg_path,'objects',args.file)).read()
    node.get_logger().info(f'type: {args.type}')
    xml = xml.replace('[MATERIAL]','Gazebo/Green' if args.type == 'prey' else 'Gazebo/Red')
    xml = xml.replace('[PATH-TO-PKG]',pkg_path)
    node.get_logger().info(xml)
    request.xml = xml

    request.robot_namespace = args.name
    node.get_logger().info(f'x: {args.x}({type(args.x)}) y: {args.y}({type(args.y)}) z: {args.z}({type(args.z)})')
    request.initial_pose.position.x = args.x
    request.initial_pose.position.y = args.y
    request.initial_pose.position.z = args.z

    future = client.call_async(request)
    
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info(
            'Result of spawn_entity: for %s' %
            (request.name,))
    else:
        node.get_logger().info('Service call failed %r' % (future.exception(),))
    node.destroy_node()
    rclpy.shutdown()


# ros2', 'run', 'gazebo_ros', 'spawn_entity.py -file /home/henri/project/src/thymio_description/thymio/urdf/thymio.sdf/, '-entity', 'robot