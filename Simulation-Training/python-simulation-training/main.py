import time
import numpy as np
import random
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
from pynput import keyboard
import cv2_utils 


import AutoEncoder.ae as ae
import AutoEncoder.model_utils as mu


import cv2

import gym_env.env_utils as env_utils
from gym_env.basic_predator_prey_gym import BasicPredatorPreyEnv

PREY_NUM = 1
PREDATOR_NUM = 3
speed = 0.2
TURN = np.array([speed,-speed])
STRAIGHT = np.array([speed,speed])

record = False
recordings_count = 0

get_random_time = lambda: random.choice(list(range(1,15)))

actions = [[0,0] for i in range(PREDATOR_NUM+PREY_NUM)]
#times = np.array([get_random_time() for i in range(PREDATOR_NUM+PREY_NUM)])

model = ae.CAE(32)
model.load_weights('AutoEncoder/checkpoints/old/ae1')

env = BasicPredatorPreyEnv(predator_num=PREDATOR_NUM,prey_num=PREY_NUM,headless=False,model='turtlebot4-2.0')

prev = time.time()
steps = 0

view = cv2_utils.CV2View('Robot Views')

    

def move(cmd):
    global actions, record
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

def on_release(key):
    pass

listener = keyboard.Listener(
    on_press=lambda key: move(key.char) if key.char is not None else None,
    on_release=on_release)

listener.start()



def main():
    global actions, steps, record, recordings_count

    while  True:
        #actions = [[0,0] for i in range(PREDATOR_NUM+PREY_NUM)]
        #times = np.array([time if time > 0 else get_random_time() for time in times])
        observations, info, done, caught = env.step(action=actions)
        
        losses = []

        for i,observation in enumerate(observations):
            #print(i,observation)
            if observation is None:
                continue
            #print(observation.shape)
            if record and steps % 12 == 0:
                cv2.imwrite('imgs/recordings/record-{}.png'.format(env_utils.number_to_n_digits(recordings_count,6)),observation)
                recordings_count += 1
        
            observation = cv2.resize(observation,(64,64))        
            preprocessed = mu.preprocess_img(observation)
            
            #Predict Model 
            latent = model.encoder(preprocessed)
            predicted = model.decoder(latent)

            # => Agent Training Here <=

            #Draw Visuals 
            view.draw_robot_view(observation,preprocessed,predicted,latent)

            # Add to Debug info
            losses.append(tf.reduce_mean(tf.keras.losses.MSE(preprocessed,predicted)).numpy())
        
        
        # A Image was create Show it 
        if view.can_show():

            # Make the image bigger
            view.resize(view.image.shape[1]*4,view.image.shape[0]*4)
            
            # If we are recording draw a red square
            if record:
                view.rectangle((0,0),(10,10),(0,0,255),-1)
            # Write the debug info
            view.debug([
                'step: {}'.format(steps),
                'record: {}'.format(record),
                'recordings_count: {}'.format(recordings_count),
                'loss: {}'.format(losses),
                'done: {}'.format(done),
            ])

            view.show()



        steps += 1
        
        #times -= 1
        #print('times',times)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
    env.stop()