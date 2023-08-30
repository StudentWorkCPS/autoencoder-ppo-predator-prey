import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np 
from AutoEncoder.ae import CAE

class Actor(tf.Model):

    def __init__(self,state_space,actions_space):
        super().__init__()
        self.model_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64,activation='relu'),
                #tf.keras.layers.Dense(64,activation='tanh',kernel_initializer=std_kernel_initilizer,bias_initializer=bias_initilizer),
                #tf.keras.layers.LayerNormalization(),
                tf.keras.layers.LSTM(64,activation='relu'),
                #tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(actions_space,activation='softmax'),
            ],name='actor'
        )
        self.model_layers.build(input_shape=(None, None,state_space))
        
        #self.log_std = tf.Variable(initial_value=tf.zeros(actions_space),trainable=True)

        #self.build(input_shape=(None, state_space))


    def call(self, input_data,action):

        actions_prob = self.model_layers(input_data)

        prob = tfp.distributions.Categorical(probs=actions_prob)
        if action is None:
            action = prob.sample([1])[0]

        return action , prob.log_prob(action)
    
    def get_trainable_values(self):
        return self.model_layers.trainable_variables #+ [self.log_std]
    
    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False, layer_range=None):
        return self.model_layers.summary(line_length=line_length, positions=positions, print_fn=print_fn, expand_nested=expand_nested, show_trainable=show_trainable, layer_range=layer_range)

class Critic(tf.Model):

    def __init__(self,state_space):
        super().__init__()
        self.model_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64,activation='relu'),
                #tf.keras.layers.LayerNormalization(),
                tf.keras.layers.LSTM(64,activation=None),
                #tf.keras.layers.LayerNormalization(),
                #tf.keras.layers.Dense(64,activation='tanh',kernel_initializer=std_kernel_initilizer,bias_initializer=bias_initilizer),
                tf.keras.layers.Dense(1, activation=None)
            ],name='critic'
        )
        self.model_layers.build(input_shape=(None, None,state_space))
        #self.build(input_shape=(None, state_space))

    def call(self, input_data):
        return self.model_layers(input_data)

    def get_trainable_values(self):
        return self.model_layers.trainable_variables

    @tf.function
    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False, layer_range=None):
        return self.model_layers.summary(line_length=line_length, positions=positions, print_fn=print_fn, expand_nested=expand_nested, show_trainable=show_trainable, layer_range=layer_range)
    

def Agent():

    def __init__(self,state_space,actions_space,options = {
       
        'gamma': 0.99,
    }):
        self.actor = Actor(state_space,actions_space)
        self.critic = Critic(state_space)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.options = options

    def get_action_value(self,state):
        return self.actor(state),self.critic(state)

    def learn(self,state,action_log_prob, old_value, reward, next_state, done,discount_factor=0.99):
        new_value = self.critic(next_state)

        v_loss = tf.reduce_mean(tf.square(reward + self.options['gamma'] * new_value * (1 - done) - old_value)) * discount_factor

        advantage = reward + self.options['gamma'] * new_value * (1 - done) - old_value

        policy_loss = -action_log_prob * advantage * discount_factor

        actor_grads = self.actor_optimizer.get_gradients(policy_loss, self.actor.get_trainable_values())
        critic_grads = self.critic_optimizer.get_gradients(v_loss, self.critic.get_trainable_values())

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.get_trainable_values()))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.get_trainable_values()))

        return policy_loss, v_loss

def obs_to_state(obs,ae,steps):
    latents = ae.obs_to_latent(obs)
    return np.append(latents,[steps/100 for _ in range(latents.shape[0])],axis=1)


def train(env,agents,episode_num,predator_num,prey_num):
    autoencoder = CAE(32)
    possible_actions = [[0.2,0.2],[0.2,0.1],[0.2,0.1]]



    for episode in range(episode_num):
        observations = env.reset()
        
        actions = [[0,0] for _ in range(predator_num+prey_num)]
        last_states = np.zeros((predator_num+prey_num,10,2))
        steps = 0

        last_states[:,-1] = obs_to_state(observations,autoencoder,steps)
        done = False
        while not done:
            if steps > 10:
                for i,obs in enumerate(observations):
                    r_idx = 0 if i < predator_num else 1

                    action,action_log_prob = agents[r_idx].get_action_value(last_states[i].reshape(1,10,33))

                    act = possible_actions[action]
                    actions[i] = act
            
            next_state, info,done,caught  = env.step(actions)


            next_state = obs_to_state(next_state,autoencoder,steps)

            rewards = [0 for _ in range(predator_num+prey_num)]
            for i in range(predator_num):
                rewards[i] = info['predator'][i]['reward']
            for i in range(prey_num):
                rewards[i+predator_num] = 1 - caught[i] * 100
        
            next_states = np.roll(last_states,-1,axis=1)
            next_states[:,-1] = next_state


            for i in range(predator_num+prey_num):
                agent = agents[0 if i < predator_num else 1]
    
                agent.learn(last_states,action_log_prob,agent.critic(last_states[i].reshape(1,10,33)),reward,next_states.reshape(1,10,33),done)

            last_states = next_states
            