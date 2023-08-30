from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
import random
import numpy as np
import os

RED = (102,0,0)
GREEN = (0,102,0)


class State():
    def __init__(self,x,y,z,roll,pitch,yaw) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def copy(self):
        return State(self.x,self.y,self.z,self.roll,self.pitch,self.yaw)
    

def has_color(image,color):
    """Check if an image has a color"""
    return np.any( np.logical_and(np.logical_and(image[:,:,0] == color[0],image[:,:,1] == color[1]),image[:,:,2] == color[2]))



def create_dir_if_not_exists(dir_path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def number_to_n_digits(number,n_digits):
    """Convert a number to a string with n digits"""
    return str(number).zfill(n_digits)



def create_client_and_wait_for_service(node, service_type, service_name):
    client = node.create_client(service_type, service_name)
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info(f'service({service_name}) not available, waiting again...')
    return client

def wheel_vels_to_vel_and_angular(wheel_velocity,wheel_distance=0.0935,wheel_radius=0.044):
    """Convert wheel velocity to differential drive velocity"""
    v = (wheel_velocity[0] + wheel_velocity[1]) / 2
    w = (wheel_velocity[1] - wheel_velocity[0]) / wheel_distance

    return [v,w]


def wheel_vels_to_Twist(wheel_velocity,wheel_distance=0.0935):
    """Convert wheel velocity to Twist message"""
    twist = Twist()
    v,w = wheel_vels_to_vel_and_angular(wheel_velocity,wheel_distance)
    #w = w if w < 1 and w > -1 else 1.0 if w > -1 else -1.0 
    #print(f'speed_v: {v} anlge_w: {w}')
    twist.linear.x = v
    twist.angular.z = w
    return twist

def generate_random_vels(possilbe_vels=[0,1]):
    """Generate a random position for the robot"""
    
    x = random.choice(possilbe_vels)
    y = random.choice(possilbe_vels)
    return np.array([x,y])

def euler_to_quaternion(roll,pitch,yaw):
    """Convert euler angles to quaternion"""
    # roll (x
    # pitch (y)
    # yaw (z)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
    return np.array([qx, qy, qz, qw])

def quaternion_to_euler(qx,qy,qz,qw):
    """Convert quaternion to euler angles"""
    # roll (x
    # pitch (y)
    # yaw (z)
    roll = np.arctan2(2*(qw*qx + qy*qz),1-2*(qx*qx + qy*qy))
    pitch = np.arcsin(2*(qw*qy - qz*qx))
    yaw = np.arctan2(2*(qw*qz + qx*qy),1-2*(qy*qy + qz*qz))
 
    return np.array([roll, pitch, yaw])

def pose_to_state(pose,state_replace : State = None):
    """Convert Pose message to State object"""
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w
    roll,pitch,yaw = quaternion_to_euler(qx,qy,qz,qw)

    if state_replace is not None:
        state_replace.x = x
        state_replace.y = y
        state_replace.z = z
        state_replace.roll = roll
        state_replace.pitch = pitch
        state_replace.yaw = yaw
        return state_replace
    
    return State(x,y,z,roll,pitch,yaw)


def twist_to_vels(twist:Twist):
    """Convert Twist message to wheel velocity"""
    v = twist.linear.x
    w = twist.angular.z
    wheel_distance = 0.0935
    wheel_radius = 0.044
    wheel_velocity = np.array([v - w * wheel_distance / 2, v + w * wheel_distance / 2]) / wheel_radius
    return wheel_velocity

def dist(a:State,b:State):
    """Calculate the distance between two states"""
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
