
# Global
from importlib import reload
import numpy as np
import tensorflow as tf
import datetime
import time
import cv2
import argparse
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from pynput import keyboard
from cv2_utils import CV2View

# Local
import agent
from AutoEncoder import ae, model_utils as mu

from gym_env.basic_predator_prey_gym import BasicPredatorPreyEnv, env_utils

additional_infos = {
    "reward_predator_function": "-prey_alive + predator_caught_prey * 100 + predator one meter away from prey * 10",
    "reward_prey_function": "alive - caught_prey * 100",
    "action_space": "3 actions: straight, turn left, turn right",
    "changes": "Added visualization for encoder",
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Environment Variables
    parser.add_argument('--predator-num', type=int, default=3)
    parser.add_argument('--prey-num', type=int, default=1)
    parser.add_argument('--headless', type=bool, default=False)

    # Training Variables
    parser.add_argument('--checkpoint', type=int, default=None)

    parser.add_argument('--total-episodes', type=int, default=1000)
    #                                                         6 FPS * 30 seconds * 4 tries  
    parser.add_argument('--episode-length', type=int, default=6 * 30  * 4)
    parser.add_argument('--minibatch-size', type=int, default=128 // 4)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--anneal-lr', type=bool, default=True)
    parser.add_argument('--log-dir', type=str, default="logs/fit/predator-prey/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--time-steps', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.998)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--vf-coef',type=float, default=0.5,
            help="coefficient of the value function")
    parser.add_argument('--ent-coef',type=float, default=0.01,
            help="coefficient of the entropy")
    parser.add_argument('--max-grad-norm',type=float, default=2,
            help="the maximum norm for the gradient clipping")

    # Model Variables
    parser.add_argument('--show-encoder-visuals', type=bool, default=False,
                        help="Show the encoder")

    return parser.parse_args()


def reward(observation, caught ,info):
    rewards = []
    # Reward Function

    prey_state = info['states'][args.predator_num]
    for i in range(args.predator_num):
        green = np.array([0,102,0])/255
        eps = 1e-6
        #print(observation,len(observation),i)
        img = observation[i]
        #prey_area = np.sum(np.logical_and(green - eps <= img,img <= green + eps))
        #prey_r = np.sqrt(prey_area)
        in_dist_reward = 0

        if env_utils.dist(prey_state, info['states'][i]) < 1:
            in_dist_reward = 10
    

        rewards.append(-1 + caught * 1000 * (info['predator_caught'] == i) + in_dist_reward) 
    
    # Reward for being alive
    for i in range(args.prey_num):
        rewards.append(1 - caught * 100)
    
    return rewards

# Code From https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/math_util.py
def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

args = parse_args()
# Debugging / Testing
file_writer = file_writer = tf.summary.create_file_writer(args.log_dir + "/metrics")
file_writer.set_as_default()


tf.summary.text(
    "infos/hyperparameters",
    "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
    ),
    step=0
)



visualize = False
view = CV2View('Encoder')
def on_press(key):
    global visualize
    
    try:
        print('{0} pressed'.format(
        key.char))
        if key.char == 'v':
            visualize = not visualize

        if not visualize:
            view.close()


        print(visualize)
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
        

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)

listener.start()

autoencoder = ae.CAE(32)
autoencoder.load_weights('AutoEncoder/checkpoints/old/ae1')

agent_options = {
    'clip_ratio': args.clip_ratio,
    'entrop_coef': args.ent_coef,
    'clip_grad': args.max_grad_norm,
    'lr': args.learning_rate,
    'value_coef': args.vf_coef,
}

turn_r = np.array([0.2,0.1])
turn_l = np.array([0.1,0.2])
straight = np.array([0.2,0.2])

possible_actions = np.array([straight, turn_r, turn_l])

action_space = len(possible_actions)
state_space = 32 + 1
# latentspace (32) + time (2) / 3 Actions
predator = agent.Agent(state_space,action_space,options=agent_options)
prey = agent.Agent(state_space,action_space,options=agent_options)

if args.checkpoint is not None:
    predator.load('models/predator_{}'.format(args.checkpoint))
    prey.load('models/prey_{}'.format(args.checkpoint))



tf.summary.text(
    "models/summary_actor",
    mu.summarize_model(predator.actor),
    step=0
)

tf.summary.text(
    "models/summary_critic",
    mu.summarize_model(predator.critic),
    step=0
)


tf.summary.text(
    "infos/additional_infos",
    "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in additional_infos.items()])
    ),
    step=0
)
    

robot_num = args.predator_num + args.prey_num
log_probs = np.zeros((robot_num,args.episode_length,1))
states = np.zeros((robot_num,args.episode_length,state_space))
actions = np.zeros((robot_num,args.episode_length,1))
# +1 because bootstrap values
dones = np.zeros((args.episode_length + 1,1))
caughts = np.zeros((args.episode_length + 1,1))
values = np.zeros((robot_num,args.episode_length + 1,1))


start_time = time.time()

env = BasicPredatorPreyEnv(args.predator_num, args.prey_num, args.headless)

start_episode = 0 if args.checkpoint is None else args.checkpoint + 1

max_latent = -10000
min_latent = 1000000

for episode in range(start_episode,args.total_episodes):

    env.unpause()

    new_lr = args.learning_rate
    if args.anneal_lr:
        new_lr = predator.anneal_lr(episode, args.total_episodes)
        
        prey.anneal_lr(episode, args.total_episodes)
        

    observations = env.reset()
    #print(observations)
    info = {}
    rewards =  np.zeros((robot_num,args.episode_length,1))
    done = False
    caught = False
    time_from_last_done = 0

    for i in range(args.episode_length):
        #observations = state.reshape(1,4)
        dones[i] = done
        caughts[i] = caught
        obs = []

        for r_idx, observation in enumerate(observations):
            
            #print(observation)
            observation = cv2.resize(observation, (64,64))
            
            # Observation is a 64x64x3 image
            preprocessed = mu.preprocess_img(observation)
            obs.append(preprocessed)
            #print(observation, observation.shape)
            latent = autoencoder.encode(preprocessed)

            max_latent = max(max_latent, np.max(latent))
            min_latent = min(min_latent, np.min(latent))

            if visualize:
                predicted = autoencoder.decode(latent)
                view.draw_robot_view(observation,preprocessed,predicted,latent)
            #print('Latent',latent.shape, latent)
                                                        # Max-time-steps = 30 * 6
            new_state = np.append(latent, [time_from_last_done/(6 * 30)])

            states[r_idx][i] = new_state

        
            
            current_state = states[r_idx][i-args.time_steps+1:i+1]
            #print(current_state.shape, current_state)
            if(i < args.time_steps):
                #print("Not enough time steps")
                continue
            
            agent_ = predator if r_idx < args.predator_num else prey
            action, log_prob, _ , value = agent_.get_action_value(current_state.reshape(1,args.time_steps,state_space))
            actions[r_idx][i] = action
            log_probs[r_idx][i] = log_prob
            values[r_idx][i] = value

        
    
        # Discrete action space to continuous action space
        act = possible_actions[actions[:,i].reshape(-1).astype(int)]
        # First 10 Steps just don't move
        act = [[0.,0.] for _ in range(robot_num)] if i < args.time_steps else act
        act[args.predator_num:] = [[0.,0.] for _ in range(args.prey_num)]

        next_observations, next_info, done, caught = env.step(act)    
        # Set the reward
        # Observation is a 64x64x3 image
        rewards[:,i] = np.array(reward(obs, caught,next_info)).reshape(-1,1)

        #for r_idx, r in enumerate(reward_values):
        #    rewards[r_idx][i] = r

        if done:
            print('Done',done,caught, np.sum(dones),np.sum(caughts))
            next_observation = env.reset()

        time_from_last_done = time_from_last_done + 1 if done else 0

        observations = next_observations
        info = next_info


        if visualize and view.can_show():
            view.resize(800, 800)
            view.debug([
                '------- Overview -------',
                'episode: {}'.format(i),
                'time: {:.2f}s'.format(time.time() - start_time),
                
                '------- Episode -------',
                'steps: {}'.format(i),
                'lr: {}'.format(new_lr),
                '------- Agents -------',
                'max_latent: {:.2f}'.format(max_latent),
                'min_latent: {:.2f}'.format(min_latent),
            ])

            view.debug_table([
                ['Agent #', 'Action', 'Prob', 'Value', 'Reward'],
            ] + [
                ['Predator {}'.format(j),str(actions[j][i][0]), '{:.2f}%'.format(float(np.exp(log_probs[j][i]) * 100)) , '{:.2f}'.format(float(values[j][i][0])), '{:.2f}'.format(float(rewards[j][i][0]))]
                for j in range(args.predator_num)
            ] + [
                ['Prey {}'.format(j), str(actions[j][i][0]), '{:.2f}%'.format(float(np.exp(log_probs[j][i]) * 100)) , '{:.2f}'.format(float(values[j][i][0])), '{:.2f}'.format(float(rewards[j][i][0]))]
                for j in range(args.predator_num,robot_num)
            ])
            #print('... Showing ...')
            
            view.show()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    env.pause()

    # Bootstrap value
    for r_idx in range(robot_num):
        observation = cv2.resize(observations[r_idx], (64,64))
        observation = mu.preprocess_img(observation)
        latent = autoencoder.encode(observation)
        new_state = np.append(latent, [time_from_last_done/(6 * 30)]).reshape(1,state_space)

        agent_ = predator if r_idx < args.predator_num else prey
        #print(states[r_idx][-args.time_steps+1:].shape, new_state.shape)
        current_state = np.append(states[r_idx][-args.time_steps+1:],new_state,axis=0)
        #print(current_state.shape, current_state)
        values[r_idx][args.episode_length] = agent_.critic(current_state.reshape(1,args.time_steps,state_space))


    dones[args.episode_length] = done
    
    states,actions,returns,advantages = agent.preprocess_data(states,actions,rewards,dones,values,args.gamma,args.lam)
    print('Return and Advantage Shape',returns.shape, advantages.shape)
    for i in range(robot_num):
        returns[i] = (returns[i] - np.mean(returns[i])) / (np.std(returns[i]) + 1e-8)

    p_states = np.zeros((robot_num, args.episode_length - args.time_steps + 1, args.time_steps, state_space))
    

    # Ignore 10 ten timesteps 
    for j in range(robot_num):
        #[[0,0],[1,1],[2,2]] - time_steps=2 -> [[[0,0],[1,1]],[[1,1],[2,2]]]
        p_states[j] = np.array([states[j,i-args.time_steps:i] for i in range(args.time_steps, args.episode_length+1)])
    
    cut_values = values[:,:-1]
    p_actions = actions[:,args.time_steps-1:]
    p_returns = returns[:,args.time_steps-1:]
    p_advantages = advantages[:,args.time_steps-1:]
    p_log_probs = log_probs[:,args.time_steps-1:]
    p_values = cut_values[:,args.time_steps-1:]

    #print(p_states.shape, p_actions.shape, p_returns.shape, p_advantages.shape, p_log_probs.shape, p_values.shape)

    clip_fracs = []

    b_idxs = np.arange(args.episode_length - args.time_steps + 1)
    
    losses = np.zeros((2, args.update_epochs,6))
    for epochs in range(args.update_epochs):
        np.random.shuffle(b_idxs)
        for r_idx in range(robot_num):
            agent_ = predator if r_idx < args.predator_num else prey
            a_idx = 0 if r_idx < args.predator_num else 1
            for i in range(0, args.episode_length, args.minibatch_size):
                
                mb_idxs = b_idxs[i:i+args.minibatch_size]
                mb_states = p_states[r_idx][mb_idxs]

                mb_actions = p_actions[r_idx][mb_idxs]
                
                mb_log_probs = p_log_probs[r_idx][mb_idxs]
                mb_returns = p_returns[r_idx][mb_idxs]
                mb_advantages = p_advantages[r_idx][mb_idxs]
                # Normalizing the advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                mb_values = p_values[r_idx][mb_idxs] 

                #print('All Shapes',mb_states.shape,mb_actions.shape,mb_log_probs.shape,mb_returns.shape,mb_advantages.shape,mb_values.shape)

                loss, v_loss, pg_loss, entropy_loss, approx_kl ,clip_frac = agent_.learn(mb_states,mb_actions,mb_log_probs,mb_returns,mb_advantages,mb_values)
                losses[a_idx][epochs] = np.array([loss, v_loss, pg_loss, entropy_loss, approx_kl ,clip_frac])

            
            
    # Explained Variance for the value function    
    # Cut bootstrapped value
    

    s = 0
    for i in range(2):
        robot_name = 'predator' if i == 0 else 'prey'
        r_c = args.predator_num if i == 0 else args.prey_num
        v = cut_values[s:s+r_c].reshape(-1)
        r = returns[s:s+r_c].reshape(-1)

        s = r_c + s

        ev = explained_variance(v, r)
        tf.summary.scalar("metrics/{}/explained_variance".format(robot_name), ev, step=episode)

    # Done to Caught Ratio
    done_count = np.sum(dones)
    caught_ratio = np.sum(caughts) / done_count
    tf.summary.scalar("metrics/caught_ratio", caught_ratio, step=episode)
    tf.summary.scalar("metrics/rounds_per_episode", done_count, step=episode)
    
    losses_name = ['loss','value_loss','policy_loss','entropy_loss','approx_kl','clip_frac']
    for i in range(2):
        robot_name = 'predator' if i == 0 else 'prey'
        for j in range(6):
            loss_name = losses_name[j]
            tf.summary.scalar("losses/{}/{}".format(robot_name,loss_name), np.mean(losses[i,:,j]), step=episode)
    

    tf.summary.scalar("charts/learning_rate", new_lr, step=episode)
    tf.summary.scalar("charts/episode_length",args.episode_length / np.sum(dones), step=episode)
    #print('is any done', np.any(dones))
    
    if episode % 10 == 0:
        prey.save('models/prey_{}'.format(episode))
        predator.save('models/predator_{}'.format(episode))

    for i in range(2):
        robot_type = "predator" if i == 0 else "prey"
        r_idx = i * args.predator_num
        
        # cumulate till done
        episode_rewards = []
        
        for j in range(1, args.episode_length):
            rewards[r_idx][j] += rewards[r_idx][j-1] * (1 - dones[j])
            #print(rewards[r_idx][j])
            if dones[j]:
                episode_rewards.append(rewards[r_idx][j - 1])
                
    
        tf.summary.scalar("rewards/{}s/avg_episodic_reward".format(robot_type), np.mean(episode_rewards), step=episode)
        #tf.summary.scalar("rewards/{}s/lowest_reward".format(robot_type), lowest_reward, step=episode)
        #robot_name = robot_type + "_" + str(i)
        #for j in range(2):
        #    tf.summary.scalar("parameters/log_std_{}_{}_{}".format(robot_type,i,j), log_stds[j], step=episode)

            #loss, v_loss, pg_loss, entropy, clip_frac = (mb_states,mb_actions,mb_log_probs,mb_returns,mb_advantages,mb_values)


            
            



