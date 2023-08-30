import rclpy
from gazebo_msgs.srv import SetLinkProperties

# Create a ROS2 node
rclpy.init()
node = rclpy.create_node('material_updater')

# Create a client to call the SetLinkProperties service
client = node.create_client(SetLinkProperties, '/gazebo/set_link_properties')

# Wait for the service to become available
while not client.wait_for_service(timeout_sec=1.0):
    node.get_logger().warn('SetLinkProperties service not available, waiting...')

# Call the SetLinkProperties service to update the material properties of an object
req = SetLinkProperties.Request()
req.link_name = 'Arena::Wall_13::link'
req.material.ambient = [0.0, 0.5, 0.0, 1.0] # new ambient color of the material
req.material.diffuse = [0.0, 1.0, 0.0, 1.0] # new diffuse color of the material
future = client.call_async(req)
rclpy.spin_until_future_complete(node, future)

# Check if the service call was successful
if future.result() is not None:
    node.get_logger().info('Material properties updated successfully')
else:
    node.get_logger().warn('Failed to update material properties')
