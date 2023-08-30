import time
import tensorflow as tf
import numpy as np
import cv2
import gym_env.env_utils as env_utils
from gym_env.basic_predator_prey_gym import BasicPredatorPreyEnv

import AutoEncoder.ae as ae

import domain_randomization as dr
import cv2_utils as cv2u
import AutoEncoder.model_utils as mu
import os 


PREDATOR_NUM = 2
PREY_NUM = 1



class DataCollector:
     
    def __init__(self):
          
        self.data = {}
        self.size = 0
        

    def add(self,key,value):
     
        if key not in self.data:
            self.data[key] = []

        self.data[key].append(value)

        self.size = np.max([self.size,len(self.data[key])])

    def reset(self):
        self.data = {}
        self.size = 0

    def get(self,key):
        return self.data[key]
    
    def get_all(self):  
        return self.data
    
    def get_keys(self):
        return list(self.data.keys())
    
    def _save(self,save_path):
        print('Saving data to: ' + save_path)
        print('Data', self.data)

        with open(save_path,'wb') as f:
            keys = self.get_keys()
            # head 
            line = ''
            for key in keys:
                line += key + ';'

            line += '\n'

            for i in range(self.size):
                for key in keys:
                    line += str(self.data[key][i]) + ';'
                line += '\n'

            f.write(line.encode('utf-8'))
        
        self.reset()

def round_weird(x):
    return np.round(x *20/2) / (20/2) 


iteration = 0

import json  

pos_data = json.load(open('eval/data.json','r'))



def initial_state(idx,type,predator_num,prey_num):
            global iteration
            
            pos = np.random.uniform(-1.7,1.7,2)
            angle = np.random.uniform(-np.pi,np.pi)
            starts = []

            while (len(starts) > 0 and (np.linalg.norm(pos -starts,axis=1) < 2).all()):
                pos = np.random.uniform(-1.8,1.8,2)
            
            starts.append(pos)

            return env_utils.State(pos[0],pos[1],0.0,0.0,0.0,angle)
        
            
            x,y = 0.0,0.0

            xs = [[0.0,0.0],[4.0,1],[1,-4.0]]
            ys = [[0.0,0.0],[4.0,0.0],[0.0,-4.0]]#[iteration % 2]


            
            if type == 'predator':
                idx = 'p' + str(idx+1)
            else:
                idx = 'prey'
            print('idx',idx)
            ((x,y),angle) = pos_data[idx][iteration]

            angle = int(round(angle / 45)) % 8

            angle = np.array([0,45,90,135,180,225,270,315])[angle] + 90
            angle = np.deg2rad(angle)

            print('initial state',x,y,angle)
            
            print('------------------RESET------------------')

            '''
            if type != 'predator':
                idx += predator_num
            print('idx',idx)
            print('iteration',iteration)
            x = xs[idx][iteration + 1 % 2]
            y = ys[idx][iteration + 1 % 2]
            angle = 0.0
            '''
            
            return env_utils.State(x,y,0.0,0.0,0.0,angle)


env = BasicPredatorPreyEnv(PREDATOR_NUM,PREY_NUM,max_steps=1000,initial_state_fn=initial_state,env_num=1,model='turtlebot4')




autoencoder = ae.CAE(32,
                input_shape=(64,64,5),
                output_filters=5,
                segmentation=True,
                architecture='small'
            )
autoencoder.load_weights('AutoEncoder/checkpoints/Second-Trained/ae')

visualize_obs = True


env_running = [0]

view = cv2u.CV2View('Robot Views')

TURN = np.array([0.2,-0.2])
STRAIGHT = np.array([0.2,0.2])

possible_actions =  [[0.2,0.2],  # Forward
                     [0.1,-0.1], # Turn right
                     [-0.1,0.1],
                     [0.0,0.0]] # Turn left
possible_actions = np.array(possible_actions) 

run = True
take_photo = False

def move(cmd):
    global run,env,take_photo
    if cmd == 's':
        run = True 
        env.steps = 0
    elif cmd == 'p':
        run = False
        print('pause robot')
        env.pause()
    elif cmd == 'q':
        env.stop()
    elif cmd == 'e':
        env.reset()
    elif cmd == 'r':
        run = False
    elif cmd == 't':
        take_photo = True
    #elif cmd == 'y':

        print('stop robot')


def obs_to_actions(obs,steps):
        global autoencoder,env,view,possible_actions,predator,prey,visualize_obs, take_photo
        #print('tfversion',obs)
        preprocessed = np.zeros((len(obs),64,64,5))
        for i in range(PREDATOR_NUM + PREY_NUM):
            #print(i)
            
            #print(obs[i].shape,obs[i].dtype)
            observation = dr.segment(obs[i])
            preprocessed[i] = observation
        #print('preprocessed',preprocessed.shape,preprocessed.dtype)
        latents = autoencoder.encode(preprocessed) #.obs_to_latent(obs,True)
        #print('latents',latents.shape,latents.dtype)
        latents = np.array(latents)

        # Only for visualization proposes
        if visualize_obs:
          decoded = autoencoder.decoder(latents).numpy() @ dr.all_features[:-1]

        input = np.zeros((PREDATOR_NUM + PREY_NUM,5))
        actions = np.zeros((PREDATOR_NUM + PREY_NUM,2))
        values = np.zeros((PREDATOR_NUM + PREY_NUM,1))
        for i in range(PREDATOR_NUM + PREY_NUM):
                
                # Only for visualization proposes
                if visualize_obs:
                    #print('preprocessed',preprocessed[i].shape,preprocessed[i].dtype)
                    img_preprocessed = preprocessed[i] @ dr.all_features[:-1]
                    img_preprocessed = img_preprocessed.astype(np.uint8)
                    
                    #print('obs',obs[i].shape,obs[i].dtype)
                    
                    #print('decoded',decoded[i].shape,decoded[i].dtype)
                    #print('latents',latents[i].shape,latents[i].dtype)
                    view.draw_robot_view(cv2.resize(obs[i],(64,64)),img_preprocessed ,decoded[i],np.array([latents[i]]))
                
                
                input = np.append(latents[i],steps/(env.max_steps))
                input = tf.convert_to_tensor(input.reshape(1,33),dtype=tf.float32)

                if take_photo:
                    print('saving photo')
                    cv2.imwrite(f'eval/{current_policy}/img_{i}_{time.time()}.png',obs[i])

                action = np.zeros((2))
                action_value = np.zeros((1,3))
                if i < PREDATOR_NUM:
                    action_value = predator(input)
                    '''
                    import tf_slim as slim
                    slim.model_analyzer.analyze_vars(predator.trainable_variables, print_info=True)
                    layers = range(7)
                    print(dir(predator))
                    print(predator.graph_debug_info)
                    #print(predator.layer-0.activation)
                    for layer in layers:
                        print(layer)

                        lay = getattr(predator,f'layer-{layer}')
                        print(dir(lay))
                        print(lay.activation)

                    '''
                    
                else:
                    action_value = prey(input)
                print(action_value)
                action = action_value[0]
                values[i] = action_value[1][0]
                action = np.argmax(action)
                actions[i] = possible_actions[action] # * [1,0,0][i]

        
            
            #print(states_dic[key])
            #print(states_dic[key].shape)
        #take_photo = False
        return actions,values


actions = np.zeros((PREDATOR_NUM + PREDATOR_NUM,2))
obs = env.reset()

predator_1 = DataCollector()
predator_2 = DataCollector()

prey_1 = DataCollector()

def predator_reward(obs):
    obs = np.array(obs)
    rew = 0
    predator_caught = 0

    obs = cv2.resize(obs, (64,64))
    green_count = cv2u.count_color(obs, [0,102,0])
    #print('green_count',green_count)


                        #print('team_reward',rew)
                    
    return green_count



def prey_reward(obs):

    return 1    
     
policies_2_check = {
    'improve1': ('checkpoints/improve1/720000_predator','checkpoints/improve1/720000_prey'),
    'improve2': ('checkpoints/pretrain-predator-3real-edge-reward-improved-discincenticse-to-pair/408000_predator','checkpoints/pretrain-predator-3real-edge-reward-improved-discincenticse-to-pair/408000_prey'),
    'improve3': ('checkpoints/improve3/720000_predator','checkpoints/improve3/720000_prey')
}

current_policy = 'improve2'
currned_pol_idx = 0

predator = tf.saved_model.load(policies_2_check[current_policy][0])
prey = tf.saved_model.load(policies_2_check[current_policy][1])


def change_policy():
    global predator,prey,current_policy,policies_2_check,currned_pol_idx
    currned_pol_idx += 1

    if currned_pol_idx >= len(policies_2_check):
        print('Successfully tested all policies')
        exit() 

    current_policy = list(policies_2_check.keys())[currned_pol_idx]

    predator = tf.saved_model.load(policies_2_check[current_policy][0])
    prey = tf.saved_model.load(policies_2_check[current_policy][1])
    


start_time = time.time()
while True:


    rob_actions,values = obs_to_actions(obs,env.steps)

    if not run:
        actions = np.zeros((PREDATOR_NUM + PREDATOR_NUM,2))
    else:
        actions = rob_actions
    
    obs, info, states = env.step(actions)

    if time.time() - start_time > 60 * 2 or states[0]['done']  :
        print('saving')#
        dir = f'eval/{current_policy}/Iteration{iteration}'
        os.makedirs(dir, exist_ok=True)
        predator_1._save(f'{dir}/predator_1_{time.time()}.csv')
        predator_2._save(f'{dir}/predator_2_{time.time()}.csv')
        prey_1._save(f'{dir}/prey_1_{time.time()}.csv')
        env.env_states[0]['done'] = True
        env.reset()
        start_time = time.time()
        iteration = iteration + 1

        if iteration >= 10:
            change_policy()
            iteration = 0

    if view.can_show():
    
        view.resize(800,800)

        view.debug([
                'Steps: {}'.format(env.steps),
                'States: {}'.format(states),
                'Run: {}'.format(run),
            ])     
        
        


        view.debug_table([
                ['Agent #', 'Action', 'Value'],
            ] + [
                ['Predator {}'.format(j),str(rob_actions[j]), '{:.2f}'.format(float(values[j][0]))]
                for j in range(PREDATOR_NUM)
            ] + [
                ['Prey {}'.format(j), str(rob_actions[j]), '{:.2f}'.format(float(values[j][0]))]
                for j in range(PREDATOR_NUM,PREDATOR_NUM + PREY_NUM)
            ])
        
        img = cv2.resize(obs[0],(64,64))
        img = cv2.resize(img,(800,800))
        cv2.imshow('Other',img)

        view.show()
        if cv2.waitKey(1) & 0xFF == ord('q'):
                view.close()
                exit(0)

    for i in range(PREDATOR_NUM):
        col = predator_1 if i == 0 else predator_2
        col.add('steps',env.steps)
        col.add('time',time.time())
        col.add('x',info['states'][i].x)
        col.add('y',info['states'][i].y)
        col.add('yaw',info['states'][i].yaw)
        col.add('action',np.argwhere((rob_actions[i] == possible_actions).all(axis=1))[0])
        col.add('value',values[i][0])
        col.add('reward',predator_reward(obs[i]))
        

    for i in range(PREY_NUM):
        idx = i + PREDATOR_NUM
        col = prey_1
        col.add('steps',env.steps)
        col.add('time',time.time())
        col.add('x',info['states'][idx].x)
        col.add('y',info['states'][idx].y)
        col.add('yaw',info['states'][idx].yaw)
        col.add('action',np.argwhere(rob_actions[idx] == possible_actions).all(axis=1)[0])
        col.add('value',values[idx][0])
        col.add('reward',prey_reward(obs[i]))
        



