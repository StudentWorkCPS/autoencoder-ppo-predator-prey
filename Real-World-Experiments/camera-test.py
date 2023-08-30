from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import LaserEcho 
from rclpy.node import Node
from cv_bridge import CvBridge
import rclpy 
from threading import Thread

from std_srvs.srv import Trigger

import matplotlib

import cv2
import numpy as np
import time
from pynput import keyboard

import argparse 

import matplotlib.pyplot as plt

CV_Bridge = CvBridge()

parser = argparse.ArgumentParser()
parser.add_argument('--topic',type=str,default='/turtlebot4_3/oakd/rgb/preview/image_raw')
parser.add_argument('--name',type=str,default='T3')





def plot(dims,labels):

    # Make a random plot...
    fig = plt.figure()
    
    ax = plt.gca()
    
    ax.set_ylim([0, 100])

    for i,dim in enumerate(dims):

        plt.plot(dim,label = labels[i])
        
    


    plt.legend()
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()
    
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    matplotlib.pyplot.close(fig)
    return data


class CameraSubscriber:

    def __init__(self,name,topic,exec,compressed = False) -> None:
        self.node = rclpy.create_node(f'{topic.replace("/","_")}_subscriber')
        self.compressed = compressed
        self.name = name
        if not compressed:
            self.node.create_subscription(Image,topic,self.image_raw,1)
        else: 
            self.node.create_subscription(CompressedImage,topic+'/compressed',self.image_raw,1)
        
        exec.add_node(self.node)

        self.observation = None
        self.topic = topic
        self.count = 0
        self.last_time = time.time()

        self.fps = 0#= 1 / (self.last_time - time.time())
        
        self.changed = False


    def image_raw(self,observation:CompressedImage):

        if hasattr(observation,'data'):
            if self.count == 0:
                print('Recieved Image',self.topic, '/compressed'  if self.compressed else '' )
            if self.compressed:
                image_cv = CV_Bridge.compressed_imgmsg_to_cv2(observation,desired_encoding='rgb8')
            else:
                image_cv = CV_Bridge.imgmsg_to_cv2(observation,desired_encoding='rgb8')
            #image_cv = cv2.cvtColor(image_cv,cv2.COLOR_BGR2RGB)
            #np_img = np.array(image_cv)    
            # Image has a red or green blob
            #if utils.has_color(np_img,RED) or utils.has_color(np_img,GREEN):
            #    print(f'Robot {self.robot_name} {self.index} has a red or green blob')
            #    cv2.imwrite(f'imgs/{self.robot_name}/{self.robot_name}_{utils.number_to_n_digits(self.index,9)}.png',image_cv)

            self.observation = image_cv
            self.count = self.count+1

            self.fps =  1 / (time.time() - last_time)
            self.last_time = time.time()

            self.changed = True

class ChannelSubscriber:
    def __init__(self,name,topic,exec) -> None:

        self.node = rclpy.create_node(f'{topic.replace("/","_")}_subscriber')
        self.name = name
        
        self.node.create_subscription(LaserEcho,topic,self.callback,1)

        exec.add_node(self.node)

        self.changed = False
        self.fps = 0
        self.last_time = time.time()

    def callback(self,msg:LaserEcho):
        #print("recieved message")
        print(msg.echoes)
        #print(msg.values)

        self.fps =  1 / (time.time() - last_time)
        self.last_time = time.time()

        self.changed = True



rclpy.init()
executor = rclpy.executors.MultiThreadedExecutor()

topic3 = '/turtlebot4_3/oakd/rgb/preview/image_raw'

topic2 = '/turtlebot4_2/oakd/rgb/preview/image_raw'

topic1 = '/turtlebot4_1/oakd/rgb/preview/image_raw'

services = [
    '/turtlebot4_2/oakd/start_camera',
    '/turtlebot4_1/oakd/start_camera',
    '/turtlebot4_3/oakd/start_camera'
]
serviceNode = rclpy.create_node(f'service_node_lol')

print("Starting Camera")
'''for service in services:

    cli = serviceNode.create_client(Trigger,service)

    while not cli.wait_for_service(timeout_sec=1.0):
        print(f'waiting for {service} to be ready')
        #time.sleep(1)
    

    req = Trigger.Request()

    future = cli.call_async(req)

    rclpy.spin_until_future_complete(serviceNode, future)

'''

args = parser.parse_args()
compressed  = False
css = [
        CameraSubscriber(args.name,args.topic,executor,compressed),
        #CameraSubscriber('T2',topic2,executor,compressed),
        #CameraSubscriber('T1',topic1,executor,compressed)
        #CameraSubscriber('T1','/turtlebot4_1/small_image',executor,compressed),
        #CameraSubscriber('T2','/turtlebot4_2/small_image',executor,compressed),
        #CameraSubscriber('T3','/turtlebot4_3/small_image',executor,compressed)
        #ChannelSubscriber('T1','/turtlebot4_1/small_image',executor),
        #ChannelSubscriber('T2','/turtlebot4_2/small_image',executor),
        #ChannelSubscriber('T3','/turtlebot4_3/small_image',executor)
      ]
FPSs = []
Unqiue_FPSs = [[] for cs in css]

executor_thread = Thread(target=executor.spin, daemon=True)
executor_thread.start()

last_imgs = [None for cs in css]
last_time = time.time()

recordings = 0

def move(cmd):
    global actions, recordings
    #print('move',cmd)
    try:  
        if cmd == 'p':
          #cv2.imwrite('imgs/photos/record-{}.png'.format(recordings),css[1].observation)
          recordings = recordings + 1
    except Exception as e:
        print(e)

def on_release(key):
    pass

listener = keyboard.Listener(
    on_press=lambda key: move(key.char) if hasattr(key,'char') else None,
    on_release=on_release)

listener.start()

import math
def round_to_digs(val,to_digs):
    powder = math.pow(10,to_digs)
    return int(val * powder) / powder

print('Waiting on Messages')
start_time = time.time()
while start_time + 60 > time.time():
    try:
        both_changed = True 
        for i in range(len(last_imgs)):
            if not css[i].changed:
                both_changed = False
                break


                
        if not both_changed: continue

        for i in range(len(last_imgs)):
            css[i].changed = False
            
        result = None 

        names = ['TOTAL']
        for i in range(len(last_imgs)):
                
                Unqiue_FPSs[i].append(css[i].fps)
                names.append(css[i].name)
                if css[i].observation is not None:
                    last_imgs[i] = css[i].observation
                    
                    if result is None:
                        result = css[i].observation
                    else: 
                        #print(result.shape,css[i].observation.shape)
                        result = np.concatenate((result,css[i].observation),axis=1)
                    


        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        
        #avg_fps = (Unqiue_FPSs[1][-1] + Unqiue_FPSs[0][-1] + Unqiue_FPSs[2][-1]) / 3
        avg_fps = 0
        print(f'FPS {round_to_digs(fps,3)} AVG {round_to_digs(avg_fps,3)}')
        
        FPSs.append(fps)
        door_size = 20
        #if len(FPSs) > door_size:
        #        FPSs = FPSs[1:door_size]
        #        Unqiue_FPSs = [arr[1:door_size] for arr in Unqiue_FPSs]
            
        dims = [FPSs] + Unqiue_FPSs
        #pypl = plot(dims,names)
        #pypl = cv2.resize(pypl,(250,250))
        
        if result is not None:
            #result = cv2.resize(result,(250*3,250),interpolation=cv2.INTER_NEAREST)
            
            #result = np.concatenate((result,pypl),axis=1)
            #result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
            pass
            #cv2.imshow('TestWindow',result)
        else: 
            #pypl = cv2.cvtColor(pypl,cv2.COLOR_RGB2BGR)
            #cv2.imshow('TestWindow',pypl)
            pass
            
        

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    except Exception as e:
        print(e)
        #break

dims = [FPSs] + Unqiue_FPSs
names = ['TOTAL'] + [cs.name for cs in css]

obj = {
    'FPSs':dims,
    'Labels':names
}

import json

fileName = '-'.join(names[1:]) + '.json'
with open(fileName, 'w') as fp:
    json.dump(obj, fp)

#cv2.destroyAllWindows()





        
'''

image = np.ones((250,250,1)) * 255

# Window name in which image is displayed
window_name = 'image'
  
# Using cv2.imshow() method
# Displaying the image
cv2.imshow(window_name, image)
  
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows()
'''