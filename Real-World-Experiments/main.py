
import time
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from pynput import keyboard
import cv2_utils 
import segment
import segment2

import AutoEncoder.ae as ae
import AutoEncoder.model_utils as mu


import cv2

import gym_env.env_utils as env_utils
from gym_env.basic_predator_prey_gym import BasicPredatorPreyEnv
from gym_env.basic_predator_prey_gym import Config


from ray.rllib.policy.policy import Policy


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default="predator")
parser.add_argument('--policy', type=str, default="Policies/440k-pretrained/440000_predator")
parser.add_argument('--model', type=str, default="AutoEncoder/checkpoints/segment/ae")
parser.add_argument('--num', type=int, default=1)

args = parser.parse_args()

PREDATOR_NUM = 1 if args.type == 'predator' else 0
PREY_NUM = 1 if args.type == 'prey' else 0

idx = args.num

speed = 0.1
TURN = np.array([speed,-speed])
STRAIGHT = np.array([speed,speed])

record = False
recordings_count = 0

get_random_time = lambda: random.choice(list(range(1,15)))

actions = [[0,0]]

#times = np.array([get_random_time() for i in range(PREDATOR_NUM+PREY_NUM)])

#model = ae.CAE(32)
model = ae.CAE(32,(64,64,5),5,True,'small')
model.load_weights(args.model)

policy = tf.saved_model.load(args.policy)




def to_namespace(type,i):
    return f'turtlebot4_{idx}'
    
    if type == 'predator':
        return f'turtlebot4_{i+idx+1}'
    
    return f'turtlebot4_{i+PREDATOR_NUM + idx + 1}'

env = BasicPredatorPreyEnv(predator_num=PREDATOR_NUM,prey_num=PREY_NUM,headless=False,real_world=True,env_num=1,custom_config=Config(robot_namespaces=to_namespace,camera_topic='/oakd/rgb/preview/image_raw'))

prev = time.time()
steps = 0

view = cv2_utils.CV2View('Robot Views')

self_control = False

values = []

def move(cmd):
    global actions, record, self_control,steps, values
    print('!!!!!!!!!!!!move!!!!!!!!!!!!!!!!!!!',cmd)
    if cmd == 'a':
        actions[0] = -TURN
    elif cmd == 'd':
        actions[0] = TURN
    elif cmd == 'w':
        actions[0] = STRAIGHT
    elif cmd == 's':
        actions[0] = -STRAIGHT
    elif cmd == 'q':
        env.stop()
    elif cmd == 'e':
        env.reset()
    elif cmd == 'r':
        print('stop robot')
        actions[0] = [0,0]
    elif cmd == 't':
        record = not record
    elif cmd == 'ö':
        self_control = True
        values = [0.0]
        steps = 0
    elif cmd == 'ä':
        self_control = False
        
        actions[0] = [0,0]

def on_release(key):
    pass

listener = keyboard.Listener(
    on_press=lambda key: move(key.char) if key.char is not None else None,
    on_release=on_release)

listener.start()


possible_actoins = np.array([
    [speed,speed],  # Forward
    [speed,-speed], # Right
    [-speed,speed], # Left
])


def main():
    global actions, steps, record, recordings_count,values

    while  True:
        steps = 0
        #actions = [[0,0] for i in range(PREDATOR_NUM+PREY_NUM)]
        #times = np.array([time if time > 0 else get_random_time() for time in times])
        #print(actions)

        env.env_states[0]['done'] = False
        observations, info, done = env.step(action=actions)

    
        
        losses = []

        for i,observation in enumerate(observations):
            #print(i,observation)
            if observation is None:
                continue
            #print(observation.shape)
            if record and steps % 12 == 0:
                cv2.imwrite('imgs/real_world/record-{}.png'.format(env_utils.number_to_n_digits(4+recordings_count,6)),observation)
                record = False
                recordings_count += 1

            print(observation.shape,observation.dtype,observation.max(),observation.min())
            segmented,segemented_img,mask,edges,in_range,remember = segment2.segment3(observation,DEBUG=True)
            #    
            observation = cv2.resize(observation,(64,64))       

            preprocessed = mu.preprocess_img(observation)
            
            #Predict Model 
            observation = cv2.cvtColor(observation,cv2.COLOR_BGR2RGB)
            observation = observation.astype(np.float32)/255.0
            
            latent = model.encoder(segmented.reshape(1,64,64,5)) 
            predicted = model.decoder(latent)


            input_dims = np.append(latent,(steps%500)/500)
            #print(input_dims.reshape(1,33).shape)
            #print(policy.summary())
            #print(policy)
            input_dims = tf.convert_to_tensor(input_dims.reshape(1,33),dtype=tf.float32)
            actions_values = policy(input_dims)
            actions_values = actions_values
            action = actions_values[0]
            value = actions_values[1]

            #print(action)
            #print(action.shape)
            action = np.argmax(action,axis=1)

            # simulate 10 fps 

            #time.sleep(0.1)


            if self_control:
                actions[0] = possible_actoins[action[0]]

            values = np.append(values,value.numpy()[0][0])
            #print(values)

            result = predicted.reshape(-1,5)
            #result = np.argmax(result,axis=1)
            fake_features = np.array([[59,59,59],[102,102,102],[178,178,178],[0,255,0],[255,0,0]])
            result =  result @ fake_features
            result = result.reshape(64,64,3)

            segmented_fake = np.argmax(segmented.reshape(-1,5),axis=1)
            segmented_fake = fake_features[segmented_fake].astype(np.uint8)
            segmented_fake = segmented_fake.reshape(64,64,3)

            # => Agent Training Here <=
            print(observation.shape,segmented.shape,predicted.shape,latent.shape)
            #Draw Visuals 
            observation = (observation * 255.0).astype(np.uint8)
            observation = cv2.cvtColor(observation,cv2.COLOR_RGB2BGR)
            view.draw_robot_view(observation,segmented_fake,result,latent,values)
            #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            mask = np.concatenate([mask,mask,mask],axis=2).astype(np.uint8)
            edges = edges.reshape(64,64,1).astype(np.uint8)
            edges = np.concatenate([edges,edges,edges],axis=2).astype(np.uint8)
            print(observation.shape,segmented.shape,in_range.shape,mask.shape,edges.shape)
            
            #image = np.concatenate((observation,segmented_fake,in_range, (1-mask) * 255,edges * 255,remember),axis=1)
            

            # Add to Debug info
            losses.append(tf.reduce_mean(tf.keras.losses.MSE(preprocessed,predicted)).numpy())
        
        
        # A Image was create Show it 
        if view.can_show():

            # Make the image bigger
            view.resize(view.image.shape[1]*4,view.image.shape[0]*4)
            
            # If we are recording draw a red square
            if record:
                view.rectangle((0,0),(10,10),(255,0,0),-1)

            if self_control:
                view.rectangle((0,0),(10,10),(0,255,0),-1)
            # Write the debug info
            view.debug([
                'fps: {}'.format(env.fps),
                'step: {}'.format(steps),
                'time: {}'.format((steps%500)/500), 
                'record: {}'.format(record),
                'recordings_count: {}'.format(recordings_count),
                'loss: {}'.format(losses),
                'done: {}'.format(done),
            ])

            print(len(values))
            view.debug_table([ 
                ['Agents','Real Action','Action','Value'],
                ['Predator',str(actions[0]),str(action),'{:.2}'.format(values[-1])],
            ])

            #image = cv2.resize(image,(image.shape[1]*4,image.shape[0]*4))
            #image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            #cv2.imshow('image',image)
            #view.image = np.array([])
            view.show()



        steps += 1
        
        #times -= 1
        #print('times',times)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
    env.stop()
