import rclpy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import LaserEcho
from typing import Union
import argparse 
import time
import segment as segment
import cv2 
from cv_bridge import CvBridge

import numpy as np


CVBRIDGE = CvBridge()
rclpy.init()



parser = argparse.ArgumentParser()
parser.add_argument('-namespace',type=str,default='turtlebot4_1')
parser.add_argument('-compressed',type=bool,default=False)
parser.add_argument('-publish-img',action='store_true')
parser.add_argument('-time',type=int,default=10)

args = parser.parse_args()

node = rclpy.create_node('camera_small_image_node')

if(args.publish_img):
    publisher = node.create_publisher(Image,args.namespace+'/small_image',10)
else: 
    publisher = node.create_publisher(LaserEcho,args.namespace+'/small_image',10)



last_time = time.time()

time_before = time.time()
avg_fps = 1

def recieved_img(mesg:Union[Image,CompressedImage]):
    global last_time
    if hasattr(mesg,'data'):
        print('recieved message')

        img = None
        if args.compressed:
            img = CVBRIDGE.compressed_imgmsg_to_cv2(mesg,'rgb8')
        else:
            img = CVBRIDGE.imgmsg_to_cv2(mesg,'rgb8')

        if args.publish_img:
            print("Publishing image")
            img = cv2.resize(img,(64,64),interpolation=cv2.INTER_NEAREST)
            result = CVBRIDGE.cv2_to_imgmsg(img,'rgb8')

            publisher.publish(result)
        else:
            print("Publishing channel")
            ints = list((np.random.random_sample(31,) * 255).astype(np.uint8))
            channel = LaserEcho()
            channel.echoes = ints 

            publisher.publish(channel)
            #channel.name = "Latentspace vector"

        
        

        s = segment.segment(img)
        fps = 1 / (time.time() - last_time)
        avg_fps = 0.9 * avg_fps + 0.1 * fps
        last_time = time.time()
        if time.time() - time_before > args.time:
            print('FPS:',fps, ' Avg FPS:',avg_fps)
            time_before = time.time()   
     
message_tpye = Image if not args.compressed else CompressedImage 
topic = args.namespace + '/oakd/rgb/preview/image_raw' + ('/compressed' if args.compressed else '')

node.create_subscription(message_tpye,topic,recieved_img,1)

print('Waiting on messages on',topic )
rclpy.spin(node)



