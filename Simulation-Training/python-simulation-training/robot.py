# Global imports

import cv2
import numpy as np

# ROS2 imports
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge

# Local imports
import utils

CVBRIDGE = CvBridge()


class Robot():
    def __init__(self,ros_node:Node,robot_name:str,inital_state:utils.State) -> None:
        
        self.robot_name = robot_name
        self.initial_state = inital_state

        self.twist_publisher = ros_node.create_publisher(Twist,robot_name+'/cmd_vel',1)

        utils.create_dir_if_not_exists(f'imgs/{robot_name}')

        self.camera_subscriber = ros_node.create_subscription(Image,robot_name+'/camera/image_raw',self.image_raw_callback,1)

        self.observation = None
        self.index = 0

    def publish_vels(self,vels:list[float]):
        self.twist_publisher.publish(utils.wheel_vels_to_Twist(vels))
    

    def image_raw_callback(self,observation:Image):
        print(f'Robot {self.robot_name} {self.index} received image')
        if hasattr(observation,'data'):
            image_cv = CVBRIDGE.imgmsg_to_cv2(observation,desired_encoding='passthrough')
            image_cv = cv2.cvtColor(image_cv,cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'imgs/{self.robot_name}/img{utils.number_to_n_digits(self.index,9)}.png',image_cv)

            self.observation = image_cv

            self.index += 1
        


        #print(observation)

    def get_initial_state(self):
        modelState = SetEntityState.Request()
        modelState.state.name = self.robot_name
        modelState.state.pose.position.x = self.initial_state.x
        modelState.state.pose.position.y = self.initial_state.y
        modelState.state.pose.position.z = self.initial_state.z


        [qx,qy,qz,qw] = utils.euler_to_quaternion(self.initial_state.roll,self.initial_state.pitch,self.initial_state.yaw)

        modelState.state.pose.orientation.x = qx
        modelState.state.pose.orientation.y = qy
        modelState.state.pose.orientation.z = qz  
        modelState.state.pose.orientation.w = qw
        
        return modelState



        