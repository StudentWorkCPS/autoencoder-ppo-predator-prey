from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gymnasium as gym
import cv2
import time
import argparse
import os

import tensorboard

from ray import tune
from ray import air

import ray.rllib.agents.ppo as ppo
from ray.rllib.algorithms.algorithm import Algorithm

from ray.tune.logger import pretty_print

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from ray.util import inspect_serializability



from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



# Custom
from gym_env.basic_predator_prey_gym import BasicPredatorPreyEnv, env_utils
import cv2_utils as cv2u


import AutoEncoder.model_utils as mu
import AutoEncoder.ae as ae
import domain_randomization as dr

import json
import torch


#autoencoder = ae.CAE(32)
#autoencoder.load_weights('AutoEncoder/checkpoints/old/ae1')

class CAE(tf.keras.Model):
    def __init__(self, latent_dim, input_shape=(64,64,1),output_filters=1,segmentation=False,architecture='default'):
        super(CAE, self).__init__()
        self.latent_dim = latent_dim
        if architecture == 'default':
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=input_shape),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(
                        filters=(32), kernel_size=3, strides=(2, 2), activation='relu'),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(
                        filters=(64), kernel_size=3, strides=(2, 2), activation='relu'),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(latent_dim),
                ],name='encoder')
            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=16*16*32, activation=tf.nn.relu),
                    tf.keras.layers.Reshape(target_shape=(16, 16, 32)),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2DTranspose(
                        filters=64, kernel_size=3, strides=2, padding='same',
                        activation='relu'),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2DTranspose(
                        filters=32, kernel_size=3, strides=2, padding='same',
                    ),
                    #tf.keras.layers.BatchNormalization(),
                    # No activation
                    tf.keras.layers.Conv2D(  
                        filters=output_filters, activation='sigmoid' if not segmentation else 'softmax', kernel_size=(3,3), strides=1, padding='same'),
                    #tf.keras.layers.BatchNormalization()
                ],name='decoder'
                )
            
        elif architecture == 'small':
            
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, activation='relu'),

                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim),
                ],name='encoder')
            
            self.decoder = tf.keras.Sequential([
                 tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                 tf.keras.layers.Dense(units=8 * 8 * latent_dim, activation='relu'),
                 tf.keras.layers.Reshape(target_shape=(8, 8, latent_dim)),
                 tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=4, padding='same',
                    activation='relu'),
                 tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                 tf.keras.layers.Conv2D(
                    filters=output_filters, kernel_size=3, activation='sigmoid' if not segmentation else 'softmax', padding='same'
                 )
                ],name='decoder')

            
        

        
    def encode(self, x,training=False):
        return self.encoder(x,training=training)
        
    def decode(self, z,training=False,apply_sigmoid=False):
        logits = self.decoder(z,training=training)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
        
    '''
    @tf.function
    def call(self, x):
        return self.decode(self.encode(x))
    '''       
    #@tf.function
    #def compute_loss(self, x,y):
    #   return tf.keras.losses.MeanSquaredError()(x,self(x))
    '''
    @tf.function
    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False, layer_range=None):
        self.encoder.summary(line_length=line_length, positions=positions, print_fn=print_fn, expand_nested=expand_nested, show_trainable=show_trainable, layer_range=layer_range)
        self.decoder.summary(line_length=line_length, positions=positions, print_fn=print_fn, expand_nested=expand_nested, show_trainable=show_trainable, layer_range=layer_range)
    '''
    
    def load(self):
        self.load_weights('AutoEncoder/checkpoints/ae')
    
        
    def load(self):
        self.load_weights('AutoEncoder/checkpoints/ae')
        
        


class MulitAgentWrapper(MultiAgentEnv):

    def __init__(self,env_config):
        super(MulitAgentWrapper, self).__init__()
        #self.env_config = env_config  = {'predator_num': 1, 'prey_num': 1, 'train_agents': ['predator', 'prey']}
        #os.system('killall -9 gazebo & killall -9 gzserver & killall -9 gzclient')
        #time.sleep(4)
        
        predator_num = env_config['predator_num']
        prey_num = env_config['prey_num']
        self.train_agents = env_config['train_agents']
        self.predator_rew = env_config['predator_rew']
        self.prey_rew = env_config['prey_rew']
        self.team_coef = env_config['team_coef']
        self.max_steps = env_config['max_steps']
        self.visualize_obs = env_config['visualize_obs']
        
        random_state = set() # set(['predator', 'prey']).difference(set(self.train_agents))
        print('random_state',random_state)

        self.env_num = env_config['env_num']

        self.starts = np.array([])

        def initial_state(idx,type,predator_num,prey_num):
            #if type == 'predator' and idx == 0:
            #    self.starts = []

            
            '''
            pos = np.random.uniform(-1.7,1.7,2)
            angle = np.random.uniform(-np.pi,np.pi)

            while (len(self.starts) > 0 and (np.linalg.norm(pos - self.starts,axis=1) < 2).all()):
                pos = np.random.uniform(-1.8,1.8,2)
            
            self.starts.append(pos)

            return env_utils.State(pos[0],pos[1],0.0,0.0,0.0,angle)
            '''
            angle = np.random.uniform(-np.pi,np.pi)
            x,y = 0.0,0.0

            if type == 'predator':
                y = np.random.uniform(0.5,1.5)
                x =  -1 + np.random.uniform(-0.5,0.5) + idx * 2

            if type == 'prey':
                y = np.random.uniform(-1,-0.5)
                x = np.random.uniform(-1.5,1.5)

            if type == 'prey':
                idx += predator_num
            #angle = idx*2*np.pi/(predator_num+prey_num)

            #x = np.sin(angle) * 1.0
            #y = np.cos(angle) * 1.0

            #angle = [-np.pi/2,np.pi,np.pi/2,0][idx]

            # Random Placing of the agent

            return env_utils.State(x,y,0.0,0.0,0.0,angle)

        self.env = BasicPredatorPreyEnv(predator_num,prey_num,max_steps=self.max_steps,initial_state_fn=initial_state,randomize_initial_state=random_state,env_num=self.env_num,model='turtlebot4')

        self.inital_state_fn = initial_state
        
        self.autoencoder = CAE(32,
                input_shape=(64,64,5),
                output_filters=5,
                segmentation=True,
                architecture='small'
            )
        self.autoencoder.load_weights('AutoEncoder/checkpoints/Second-Trained/ae')

        if env_config['actions']:
            self.possible_actions = env_config['actions']
        else:
            self.possible_actions = np.array([
                [0.2,0.2],  # Forward
                [0.2,-0.2], # Turn right
                [-0.2,0.2], # Turn left
            ])

        print('possible_actions',self.possible_actions)
        
        self.env_running = [eid for eid in range(self.env_num)]
        
        self.observation_space = gym.spaces.Box(low=-1,high=1,shape=(33, 1), dtype=np.uint8) # 32 latent + 1 step
        self.action_space = gym.spaces.Discrete(len(self.possible_actions)) # 3 possible actions
        print('action space',self.action_space)
        self.steps = 0
        self.resetted = False

        self.predator_num = predator_num
        self.prey_num = prey_num

        #self.visualize = True
        if self.visualize_obs:
            self.view = cv2u.CV2View('Robot Views')

        self.agents = ['env{}_predator{}'.format(eid,i) for i in range(self.predator_num) for eid in range(self.env_num)] + ['env{}/prey{}'.format(eid,i + self.predator_num) for i in range(self.prey_num) for eid in range(self.env_num)]
        self._agent_ids = self.agents
    #@tf.function()
    def obs_to_state(self,obs,steps):
        #print('tfversion',obs)
        preprocessed = np.zeros((len(obs),64,64,5))
        for i in range(self.predator_num + self.prey_num):
            observation = dr.segment(obs[i])
            preprocessed[i] = observation
        print('preprocessed',preprocessed.shape,preprocessed.dtype)
        latents = self.autoencoder.encode(preprocessed) #.obs_to_latent(obs,True)
        print('latents',latents.shape,latents.dtype)
        latents = np.array(latents)

        # Only for visualization proposes
        if self.visualize_obs:
          decoded = self.autoencoder.decoder(latents).numpy() @ dr.all_features[:-1]

        states_dic = {}
        for eid in self.env_running:
            for i in range(self.predator_num + self.prey_num):
                rob_type = 'predator' if i < self.predator_num else 'prey'
                key ='env'+str(eid) + '_' +  rob_type + str(i)
                
                # Only for visualization proposes
                if self.visualize_obs:
                    print('preprocessed',preprocessed[i].shape,preprocessed[i].dtype)
                    img_preprocessed = preprocessed[i] @ dr.all_features[:-1]
                    img_preprocessed = img_preprocessed.astype(np.uint8)
                    print('obs',obs[i].shape,obs[i].dtype)
                    
                    print('decoded',decoded[i].shape,decoded[i].dtype)
                    print('latents',latents[i].shape,latents[i].dtype)
                    self.view.draw_robot_view(cv2.resize(obs[i],(64,64)),img_preprocessed ,decoded[i],np.array([latents[i]]))
                
                if rob_type in self.train_agents:
                    # Normalize observation from -1 to 1
                    latents[i] = latents[i] + 150 
                    latents[i] = latents[i] / 300
                    
                    states_dic[key] = np.append(latents[i],steps/(self.env.max_steps))
            
            #print(states_dic[key])
            #print(states_dic[key].shape)
        return states_dic

    def step(self, actions_dict):
        actions = np.zeros((self.env_num,self.predator_num + self.prey_num,2))
        # Multi env test
        for key,action in actions_dict.items():
            #print(key,action)
            eid = int(key.split('_')[0].split('env')[1])
            if 'predator' in key:
                i = int(key.split('_')[1].split('predator')[1])
                actions[eid][i] = self.possible_actions[action]

            elif 'prey' in key:
                i = int(key.split('_')[1].split('prey')[1])
                actions[eid][i] = self.possible_actions[action]

        actions = actions.reshape(-1,2)
        obs, info, env_states = self.env.step(actions)

        states_dic = self.obs_to_state(obs,self.steps)        
        reward = {}
        dones = {}
        infos = {}
        #prey_states_pos = info['states'][self.predator_num]

        # Reward Calculation
        current_env_running = self.env_running.copy()
        for eid in current_env_running:
            caught = env_states[eid]['caught']
            predator_caught = env_states[eid]['predator_caught']
            done = env_states[eid]['done']
            env_offset = eid * (self.predator_num + self.prey_num)
            if 'predator' in self.train_agents:
                for i in range(self.predator_num):
                    #range_ = env_utils.dist(info['states'][i], prey_states_pos)
                    key = 'env{}_predator{}'.format(eid,i)

                    rew = 0

                    if 'count_green' in self.predator_rew:
                        ob = obs[i]
                        ob = cv2.resize(ob, (64,64))
                        green_count = cv2u.count_color(ob, [0,102,0])
                        #rint('green_count',green_count)
                        if 'over_time' in self.predator_rew:
                            green_count = green_count - env_states[eid]['steps']

                        rew += green_count

                        red_count = cv2u.count_color(ob,[104,0,0])

                        rew += -red_count / 10
                    if 'caught_reward' in self.predator_rew:
                        #print('caught',i,predator_caught == i)
                        rew += 10000 * (predator_caught == i) * caught
                
                    if 'prey_alive' in self.predator_rew:
                        rew += -1
                        #print('prey_alive',-1)

                    if 'team_reward' in self.predator_rew:
                        rew *= (1 - self.team_coef)
                        rew += self.team_coef * caught * 100000
                        #print('team_reward',rew)
                    
                    
                    # Normalize the reward
                    reward[key] = rew / 10000
                    dones[key] = done
                    infos[key] = {'state':info['states'][i + env_offset], 'predator_caught':predator_caught == i,'velocity':info['velocities'][i + env_offset]}
                                   
        
            if 'prey' in self.train_agents:
                for i in range(self.prey_num):
                    key = 'env{}_prey{}'.format(eid,i + self.predator_num)

                    rew = 0
                    if 'caught_reward' in self.prey_rew:
                        rew += -100 * caught
                        #print('prey_caught_reward',-100)
                    
                    if 'alive_reward' in self.prey_rew:
                        rew += 1
                        #print('prey_alive',1)
                    #if 'edge_reward' in self.prey_rew:
                        robot_current_state = info['states'][i + env_offset]
                        edges = 1.7
                        on_edges = (robot_current_state.x > edges or 
                                    robot_current_state.x < -edges or 
                                    robot_current_state.y > edges or 
                                    robot_current_state.y < -edges)
                        rew += -2 * int(on_edges)


                    # Normalize the reward => Easier to train the Value Function
                    reward[key] = rew / self.max_steps
                    dones[key] = done
                    infos[key] = {'state':info['states'][i+self.predator_num+env_offset],'velocity':info['velocities'][i+self.predator_num+env_offset]}
            
            if done:
                self.env_running.remove(eid)
            #reward[key] = -1 if caught else 1
        #print('dones',dones)   
        dones['__all__'] = np.all(list(dones.values()))
        #print('dones',dones.values())
        if self.visualize_obs and self.view.can_show():
            self.view.resize(800,800)
            self.view.show()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.view.close()
                self.close()
                exit(0)
        print("rewards",reward)
        #print('dones',dones)
        return states_dic, reward, dones, infos
    

    def reset(self,seed=None,options=None):
        super().reset()
        self.resetted = True

        self.steps = 0
        for i,robot in enumerate(self.env.robots):
            idx = i % self.predator_num
            type = 'predator' if i < self.predator_num else 'prey'

            robot.initial_state = self.inital_state_fn(idx,type,self.predator_num,self.prey_num)

        obs = self.env.reset()

        states = self.obs_to_state(obs,self.steps)
        #print('resetted',states['predator0'].shape)
        self.env_running = [eid for eid in range(self.env_num) if self.env.env_states[eid]['done'] == False]

        if self.visualize_obs and self.view.can_show():
            self.view.show()

        return states
    
    def close(self):
        self.env.stop()

    def cleanup(self):
        self.env.stop()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--exp_name", type=str, default="predator_prey")
    argparser.add_argument("--max_timesteps", type=int, default=720_000)
    argparser.add_argument("--predator_num", type=int, default=3
                           ,help="Number of predators")
    argparser.add_argument("--prey-num", type=int, default=1,
                           help="Number of prey")
    argparser.add_argument("--train-agent", type=str, default='["predator","prey"]',
                            help="Which agent to train")
    
    argparser.add_argument("--use-lstm",action="store_true", default=False)
    argparser.add_argument("--use-attention", action="store_true", default=False)

    argparser.add_argument("--ckpt-restore", type=str, default=None)

    argparser.add_argument("--use-wandb", action="store_true", default=False)

    argparser.add_argument("--predator-rew", type=str, default='["caught_reward","count_green"]')
    argparser.add_argument("--prey-rew", type=str, default='["caught_reward","alive_reward","edge_reward"]')
    argparser.add_argument("--team-coef", type=float, default=0.1)
    argparser.add_argument("--env-num", type=int, default=1)

    argparser.add_argument("--max_steps", type=int, default=500)

    argparser.add_argument("--actions", type=str, default=None)
    argparser.add_argument("--visualize-obs", action="store_true", default=False)

    args = argparser.parse_args()
    
    return args

args = parse_args()
print(args.predator_rew)

inspect_serializability(MulitAgentWrapper)

def policy_mapping_fn(agend_id,self,worker=None):
    #print(agend_id,'called mapping')
    return 'predator' if agend_id.find('predator') >= 0 else 'prey'


predator_weights = None
prey_weights = None

if args.ckpt_restore is not None:
    #prevoise_config = Algorithm.from_checkpoint(args.ckpt_restore,)
    # Hack to kill gazebo again 
    pass

train_agents = json.loads(args.train_agent)

algo = (ppo.PPOConfig()
    .training(
        entropy_coeff=0.01,
    )
    .framework("tf2")
    .environment(MulitAgentWrapper, env_config={
        "predator_num": args.predator_num,
        "prey_num": args.prey_num,
        "train_agents": train_agents,
        "predator_rew": json.loads(args.predator_rew),
        "prey_rew": json.loads(args.prey_rew),
        "team_coef": args.team_coef,
        "max_steps": args.max_steps,
        "actions": json.loads(args.actions) if args.actions is not None else None,
        "env_num": args.env_num,
        "visualize_obs": args.visualize_obs
    })
    .multi_agent(policies={
        "predator": (
                None,
                gym.spaces.Box(low=-1, high=1, shape=(33,), dtype=np.float32),
                None,
                {
                    "entropy_coeff": 0.01,
                    "model": {    
                        "use_lstm": args.use_lstm,
                        "use_attention": args.use_attention,
                    }
                }
            ),
        "prey": (
                None,
                gym.spaces.Box(low=-1, high=1, shape=(33,), dtype=np.float32),
                None,
                {
                    "entropy_coeff": 0.01,
                    "model": {
                        "use_lstm": args.use_lstm,
                        "use_attention": args.use_attention,
                    }
                }
            ),

        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=train_agents
    ) 
    .rollouts(num_rollout_workers=1, num_envs_per_worker=1)
)

algo = algo.build(use_copy=False)

if args.ckpt_restore is not None:

    prevoise_config = Algorithm.from_checkpoint(args.ckpt_restore)

    #algo.restore(args.ckpt_restore)
    for agent in ['predator','prey']:
        algo.get_policy(agent).set_weights(prevoise_config.get_policy(agent).get_weights())




checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', args.exp_name)


# Writers for Tensorboard    
writer = tf.summary.create_file_writer('logs/{}'.format(args.exp_name)+ time.strftime("%Y%m%d-%H%M%S", time.localtime()))
writer.set_as_default()

if args.use_wandb:
    import wandb

    wandb.init(project="predator_prey", 
               name=args.exp_name,
               save_code=True,
               sync_tensorboard=True,
               )
    wandb.config.update(args)


if args.ckpt_restore is not None and prevoise_config is not None:
    print('restore from',args.ckpt_restore)
    #algo.restore(args.ckpt_restore)

    #algo.import_model(args.ckpt_restore)
    #print('',prevoise_config.get_weights().keys())
    #algo.set_weights(prevoise_config.get_weights())

    #print(algo.get_policy("predator").get_weights().all(prevoise_config.get_policy("predator").get_weights()))


    #algo._timesteps_total = prevoise_config._timesteps_total
    #algo._iteration = prevoise_config._iteration

    algo.save(checkpoint_path)
    prevoise_config.stop()
    algo.stop()
    time.sleep(4)
    # Destroy all gazebo instances
    os.system('killall -9 gazebo & killall -9 gzserver & killall -9 gzclient')
    
    # Strt current alogrithm
    algo = Algorithm.from_checkpoint(checkpoint_path+'/checkpoint_000000')

#raise Exception('stop')
print(pretty_print(algo.config.to_dict()))


#algo.get_policy("predator").set_weights(prevoise_config.get_policy("predator").get_weights())
#algo.import_policy_model_from_h5('checkpoints/raylib-ppo-predator-prey-steps-{}/checkpoint_000129'.format(516_000),['predator'])


total_steps_max = 720_000
step = 0
while step < total_steps_max:
    result = algo.train()
    print(pretty_print(result))
    step = result['timesteps_total']
    tf.summary.text('result',pretty_print(result), step=step)
    tf.summary.scalar('rollout/ep_rew_mean', result['episode_reward_mean'], step=step)
    tf.summary.scalar('rollout/ep_len_mean', result['episode_len_mean'], step=step)
    tf.summary.scalar('episodes_total', result['episodes_total'], step=step)
    total_steps_iter = result['num_env_steps_sampled'] 
    fps = total_steps_iter / result['time_this_iter_s']
    tf.summary.scalar('time/fps', fps, step=step)

    for key in train_agents:
        max_rew= result['policy_reward_max'][key]
        tf.summary.scalar('rewards/reward_max/{}'.format(key), max_rew, step=step)
        min_rew= result['policy_reward_min'][key]
        tf.summary.scalar('rewards/reward_min/{}'.format(key), min_rew, step=step)
        mean_rew= result['policy_reward_mean'][key]
        tf.summary.scalar('rewards/reward_mean/{}'.format(key), mean_rew, step=step)



    
    #for k,v in result['info'].items():
    #    tf.summary.scalar('info/{}'.format(k), v, step=step)


    writer.flush()

    if step % 1000 == 0:
        checkpoint = algo.save(checkpoint_path)
        for key in train_agents:
            print('exporting',key)
            path = checkpoint_path+ '/{}_{}'.format(step,key)

            print(path)

            policy = algo.get_policy(key)
            #print(policy)
            #print(policy.model)
            
            #print(policy.model.base_model)


            algo.export_policy_model(path,key)
            #print(policy)
            
            #policy.export_model(path)
        print("checkpoint saved at", checkpoint)
        #exit()

        

'''
[0.2,0.2],  # Forward
                [0.2,-0.2], # Turn right
                [-0.2,0.2], # Turn left
'''
#python3 ray-lib-ppo-train.py  --actions '[[0.0,0.0],[0.1,-0.1],[-0.1,0.1]]' --use-attention --use-wandb --team-coef 0.2 --predator-rew '["caught_reward","count_green","team_reward"]' --max_steps 500 --predator_num 2 --visualize-obs --exp_name real-world-segmetation-6

# python3 ray-lib-ppo-train.py  --actions '[[0.1,0.1],[0.1,-0.1],[-0.1,0.1]]' --use-wandb --team-coef 0.2 --predator-rew '["caught_reward","count_green","team_reward"]' --max_steps 500 --predator_num 2 --visualize-obs --exp_name real-world-new-best-ae-only-predator-pretrained-4 --train-agent '["predator"]' --ckpt-restore checkpoints/real-world-new-best-ae-only-predator-turn-3/checkpoint_000124


