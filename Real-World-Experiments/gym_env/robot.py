# Global imports

import cv2
import numpy as np
import glob
import os
from threading import Thread



# ROS2 imports

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge
import rclpy

# Local imports
import gym_env.env_utils as env_utils

CVBRIDGE = CvBridge()

TO_RANDOM_POSITION = False




class Robot():
    def __init__(self,ros_node:Node,robot_name:str,inital_state:env_utils.State,executor:rclpy.executors.MultiThreadedExecutor,random_pos:bool=False,env_id:int=0, real_world:bool=False,cmd_topic='/cmd_vel',camera_topic='/camera/image_raw',namespace='predator') -> None:
        
        self.robot_name = robot_name
        self.initial_state = inital_state
        self.current_state = inital_state.copy()
        self.robot_node = rclpy.create_node(robot_name + '_node')

        print('cmd topic',cmd_topic)
        self.twist_publisher = self.robot_node.create_publisher(Twist,cmd_topic,1)

        self.random_pos = random_pos
        env_utils.create_dir_if_not_exists(f'imgs/{robot_name}')
        for file in glob.glob(f'imgs/{robot_name}/*'):
            os.remove(file)

        print('camera topic',camera_topic)
        self.camera_subscriber = self.robot_node.create_subscription(Image,camera_topic,self.image_raw_callback,1)

        self.namespace = namespace
        

        executor.add_node(self.robot_node)

        self.observation = None
        self.index = 0
        self.reseted = True
        self.new_img = False

        self.last_vels = [0,0]

        self.env_id = env_id
        self.env_offset = [env_id*4.2,0]


        if not real_world: 
            from gazebo_msgs.srv import SetEntityState
            self.SetEntityState = SetEntityState


    def __str__(self) -> str:
        return self.robot_name
    
    def __repr__(self) -> str:
        return f'{self.robot_name} ({self.namespace})'
    # TO recive the observation        
    def spin(self):
        rclpy.spin(self.robot_node)


    def publish_vels(self,vels:list[float]):
        #if self.last_vels[0] != vels[0] or self.last_vels[1] != vels[1]:
            self.twist_publisher.publish(env_utils.wheel_vels_to_Twist(vels))
            self.last_vels = vels

    def get_observation(self):
        self.new_img = False
        return self.observation

    def set_current_vels(self,vels:list[float]):
        self.last_vels = vels


    def image_raw_callback(self,observation:Image):
        #print(f'Robot {self.robot_name} {self.index} received image')
        if hasattr(observation,'data'):
            image_cv = CVBRIDGE.imgmsg_to_cv2(observation,desired_encoding='rgb8')
            #image_cv = cv2.cvtColor(image_cv,cv2.COLOR_BGR2RGB)
            #np_img = np.array(image_cv)    
            # Image has a red or green blob
            #if utils.has_color(np_img,RED) or utils.has_color(np_img,GREEN):
            #    print(f'Robot {self.robot_name} {self.index} has a red or green blob')
            #    cv2.imwrite(f'imgs/{self.robot_name}/{self.robot_name}_{utils.number_to_n_digits(self.index,9)}.png',image_cv)

            self.observation = image_cv
            self.new_img = True
            self.index += 1
        


        #print(observation)

    def get_initial_state(self):
        modelState = self.SetEntityState.Request()

        self.reseted = True

        if(self.random_pos):
            self.initial_state.x = np.random.uniform(1,1.5)
            self.initial_state.y = np.random.uniform(-1.5,1.5)
            self.initial_state.yaw = np.random.uniform(-np.pi,np.pi)


        modelState.state.name = self.robot_name
        modelState.state.pose.position.x = self.initial_state.x + self.env_offset[0]
        modelState.state.pose.position.y = self.initial_state.y + self.env_offset[1]
        modelState.state.pose.position.z = self.initial_state.z


        [qx,qy,qz,qw] = env_utils.euler_to_quaternion(self.initial_state.roll,self.initial_state.pitch,self.initial_state.yaw)

        modelState.state.pose.orientation.x = qx
        modelState.state.pose.orientation.y = qy
        modelState.state.pose.orientation.z = qz  
        modelState.state.pose.orientation.w = qw
        
        return modelState



        