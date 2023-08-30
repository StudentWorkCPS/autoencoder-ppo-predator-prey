import agent
import gym
import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
import tensorflow as tf


import datetime
from tensorboard.plugins.hparams import api as hp
import tensorboard as tb

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

tf.config.set_visible_devices([], 'GPU')

reload(agent)
import random

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()


state_space = 4

env = gym.make('CartPole-v1')
env.reset()
#env.render()
learning_rate = 2.5e-4
steps = 25000
episode_length = 128 * 4
agent1 = agent.Agent(state_space,2)
ep_rewards = []
total_avg_r = []
best_reward = -np.inf
avg_reward_list = []

minibatch_size = 128 // 4

anneal_lr = True


agent1.set_lr(learning_rate)


log_probs = np.zeros((episode_length,1))
states = np.zeros((episode_length,state_space))
actions = np.zeros((episode_length,1))
# +1 because bootstrap values
dones = np.zeros((episode_length + 1,1))
values = np.zeros((episode_length + 1,1))

for step in range(steps):

    if anneal_lr:
        agent1.anneal_lr(step,steps)

    #env = gym.make('Pendulum-v1',render_mode='rgb_array')
    state, info = env.reset()
    #print(state)
    all_loss = []
    rewards = []
    done = False
    print("new episode")
    print("generating data")
    current_reward = 0
    reset_time = 120
    for i in range(episode_length):
        state = state.reshape(1,4)
        states[i] = state
        dones[i] = done


        action , log_prob, _ , value = agent1.get_action_value(state)
        
        actions[i] = action.numpy()
        log_probs[i] = log_prob.numpy()
        values[i] = value[0].numpy()

        #act = 0 if action.numpy()[0] < 0 else 1

        next_state, reward, done , _ , info = env.step(action.numpy()[0])

        #reward = reward.reshape(1,1)
        #print(next_state)
        #if next_state[0] >= -0.4:
        #    reward += 10

        rewards.append(reward)
        
        state = next_state

        if done: #or (i + 1) % reset_time == 0:
            state,info = env.reset()

    
    state = state.reshape(1,4)
    value = agent1.critic(state)
    values[episode_length] = value[0].numpy()
    dones[episode_length] = done


    states,actions,returns,adv = agent.preprocess_data(states,actions,rewards,dones,values)
    clip_fracs = []
    b_idxs = np.arange(episode_length)
    print("training agent")
    approx_kls = []
    for epocs in range(10):
        np.random.shuffle(b_idxs)
        
        
        for batch in range(0,episode_length,minibatch_size):
            mb_idxs = b_idxs[batch:batch+minibatch_size]
            mb_adv = adv[mb_idxs]
            mb_adv = (mb_adv - np.mean(mb_adv)) / (np.std(mb_adv) + 1e-8)

            loss , v_loss, pg_loss, entropy_loss  , approx_kl , clip_fracs  = agent1.learn(states[mb_idxs],actions[mb_idxs],mb_adv,log_probs[mb_idxs],values[mb_idxs],returns[mb_idxs])
            all_loss.append(loss)
            approx_kls.append(approx_kl)
            #clip_fracs.append(clip_fracs)

    approx_kl = np.mean(approx_kls)

        #if approx_kl > 1.5 * 0.01:
        #    print("Early stopping at step ", epocs, "due to reaching max kl")
        #    break

    #print('rewards',len(rewards),rewards)
    #avg_reward = np.mean(rewards)
    total_round_returns = []
    total_episode_rewards = [0]
    for i in range(0,len(rewards)):
        total_episode_rewards[-1] += rewards[i]
        if dones[i]:
            total_round_returns.append(returns[i])
            total_episode_rewards.append(0)

    avg_episodic_return = np.mean(total_episode_rewards)
    avg_return = np.mean(returns)

    avg_loss = np.mean(all_loss)
    print("Finished: ",step,"/",steps,"avg reward: ",avg_episodic_return, "avg loss: ",avg_loss)
    

    #tf.summary.scalar('Avg_Reward', avg_reward, step=step)

    tf.summary.scalar('charts/episodic_return_test', avg_episodic_return, step=step)
    tf.summary.scalar('charts/avg_return', avg_return, step=step)
    tf.summary.scalar('charts/learning_rate', agent1.anneal_lr(step,steps), step=step)

    # Losses charts 
    tf.summary.scalar('losses/policy_loss', pg_loss, step=step)
    tf.summary.scalar('losses/value_loss', v_loss, step=step)
    tf.summary.scalar('losses/entropy', entropy_loss, step=step)
    tf.summary.scalar('losses/total_loss', loss, step=step)

    tf.summary.scalar('losses/clipfrac', np.mean(clip_fracs), step=step)
    tf.summary.scalar('losses/approx_kl', approx_kl, step=step)

    
    log_std = agent1.actor.log_std.numpy()[0]
    #print(log_std)
    tf.summary.scalar('parameters/log_std_actor', log_std, step=step)
    #tf.summary.scalar('parameters/', , step=step)

    #file_writer.add_scalar('Avg_Reward', avg_reward, step=step)
    
    #ep_rewards.append(avg_reward)
    #plt.plot(ep_rewards)
    #plt.show()
    highest_reward = max(total_episode_rewards)

    if(highest_reward > best_reward):
        best_reward = highest_reward
        agent1.actor.save_weights("best_model/actor")
        agent1.critic.save_weights("best_model/critic")
        print("saved best model")

    #state,info = env.reset()

env.close()
