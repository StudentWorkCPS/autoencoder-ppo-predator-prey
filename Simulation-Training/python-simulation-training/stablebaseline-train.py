import gym
import numpy as np
import AutoEncoder.model_utils as mu
import AutoEncoder.ae as ae
from gym_env.basic_predator_prey_gym import BasicPredatorPreyEnv, env_utils
import cv2_utils as cv2u
import cv2

import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import os




class GymWrapper(gym.Env):
    def __init__(self,predator_num, prey_num):
        self.env = BasicPredatorPreyEnv(predator_num,prey_num)
        

        self.possible_actions = np.array([
            [0.2,0.2],  # Forward
            [0.2,-0.2], # Turn right
            [-0.2,0.2], # Turn left
        ])

        self.autoencoder = ae.CAE(32)
        self.autoencoder.load_weights('AutoEncoder/checkpoints/old/ae1')


        self.observation_space = gym.spaces.Box(low=-15, high=15, shape=(33, 1), dtype=np.uint8) # 32 latent + 1 step
        self.action_space = gym.spaces.Discrete(len(self.possible_actions)) # 3 possible actions

        self.steps = 0

        self.predator_num = predator_num
        self.prey_num = prey_num

        self.visualize = True
        #self.view = cv2u.CV2View('Predator View')


    def obs_to_state(self,obs,steps):
        latents = self.autoencoder.obs_to_latent(obs)
        
        states = np.zeros((latents.shape[0],latents.shape[1]+1))
        for i in range(latents.shape[0]):
            states[i] = np.append(latents[i],steps/(self.env.max_steps)).reshape(-1)
        return states

    def step(self, action):
        
        action = [self.possible_actions[action]] + [[0,0]]
        obs, info, done, caught = self.env.step(action)

        states = self.obs_to_state(obs,self.steps)        

        reward = np.zeros(self.predator_num+self.prey_num)
    
        prey_states_pos = info['states'][self.predator_num]

        # Reward Calculation
        for i in range(self.predator_num):
            range_ = env_utils.dist(info['states'][i], prey_states_pos)


            ob = obs[i]
            ob = cv2.resize(ob, (64,64))
            '''if self.visualize:
                prepr =mu.preprocess_img(ob)
                lat = states[i][:-1].reshape(1,-1)
                pred = self.autoencoder.decode(lat)

                self.view.draw_robot_view(ob,prepr,pred,lat)'''

            green = np.array([0,102,0])
            #print(green)
            #print(ob)

            count_green = cv2u.count_color(ob, green)

            #print('Green Counted',count_green)


            reward[i] = (count_green +  caught * 1000 * (info['predator_caught'] == i) ) / 1000
        if self.visualize and self.view.can_show():
            print('showing')
            #self.view.resize(800,800)
            #self.view.show()
        #self.view.draw_robot_view()
        # Reward Calculation
        #for i in range(self.prey_num):
        #    reward[i+self.predator_num] = 1 - caught * 100 * info['prey_caught']            
        print(self.steps,done)
        return states[0].reshape(-1,1), reward[0], done, info
    

    def reset(self):
        self.steps = 0

        obs = self.env.reset()

        return self.obs_to_state(obs,self.steps)[0].reshape(-1,1)

    def close(self):
        self.env.stop()

        return super().close()
        
env = GymWrapper(1,1)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models_saved/')
#if os.path.exists('A2C_predator_prey.zip') or os.path.exists('A2C_predator_prey'):
#    model = A2C.load("A2C_predator_prey")
#    print('loaded')
#else:
#    model = A2C('MlpPolicy',env,normalize_advantage=True, n_steps=720, verbose=1, tensorboard_log="./tensorboard/a2c_predator_prey/" + time.strftime("%Y%m%d-%H%M%S",time.localtime()))
model = PPO('MlpPolicy',env,verbose=1, tensorboard_log="./tensorboard/ppo_predator_prey/" + time.strftime("%Y%m%d-%H%M%S",time.localtime()))


model.learn(
            total_timesteps=700_000,
            log_interval=1,
            tb_log_name="PPO_predator_prey",
            callback=checkpoint_callback
            )
model.save("PPO_predator_prey")

obs = env.reset()
while True:
    print('finished')
    action = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action[0])

    if done:
        obs = env.reset()