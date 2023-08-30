# Inspiration by https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

bias_initilizer = tf.keras.initializers.Constant(0.0)
kernel_initilizer = lambda std: tf.keras.initializers.Orthogonal(std)
std_kernel_initilizer = kernel_initilizer(np.sqrt(2.0))


class Critic(tf.keras.Model):
  def __init__(self,state_space):
    super().__init__()
    self.model_layers = tf.keras.Sequential(
      [
        tf.keras.layers.Dense(64,activation='tanh',kernel_initializer=std_kernel_initilizer,bias_initializer=bias_initilizer),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LSTM(64,activation=None,kernel_initializer=std_kernel_initilizer,bias_initializer=bias_initilizer),
        #tf.keras.layers.LayerNormalization(),
        #tf.keras.layers.Dense(64,activation='tanh',kernel_initializer=std_kernel_initilizer,bias_initializer=bias_initilizer),
        tf.keras.layers.Dense(1, activation=None,kernel_initializer=kernel_initilizer(1.),bias_initializer=bias_initilizer)
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
    

class Actor(tf.keras.Model):
  def __init__(self,state_space,actions_space):
    super().__init__()
    self.model_layers = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64,activation='tanh',kernel_initializer=std_kernel_initilizer,bias_initializer=bias_initilizer),
            #tf.keras.layers.Dense(64,activation='tanh',kernel_initializer=std_kernel_initilizer,bias_initializer=bias_initilizer),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LSTM(64,activation='tanh',kernel_initializer=std_kernel_initilizer,bias_initializer=bias_initilizer),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(actions_space,activation='softmax',kernel_initializer=kernel_initilizer(.01),bias_initializer=bias_initilizer),
        ],name='Actor'
    )
    self.model_layers.build(input_shape=(None, None,state_space))
    
    self.log_std = tf.Variable(initial_value=tf.zeros(actions_space),trainable=True)

    #self.build(input_shape=(None, state_space))

  def call(self, input_data):
    return self.model_layers(input_data)
  
  def get_trainable_values(self):
    return self.model_layers.trainable_variables #+ [self.log_std]
  
  @tf.function
  def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False, layer_range=None):
    return self.model_layers.summary(line_length=line_length, positions=positions, print_fn=print_fn, expand_nested=expand_nested, show_trainable=show_trainable, layer_range=layer_range)
    
  

class Agent():
    def __init__(self,state_space,action_space, options={
        'clip_ratio': 0.2,
        'entrop_coef': 0.01,
        'value_coef': 0.5,
        'lr': 2.5e-4,
        'clip_grad': 0.5
    }):
    
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4,epsilon=1e-5)

        self.actor = Actor(state_space,action_space)
        self.critic = Critic(state_space)

        self.action_space = action_space
        
        
        self.clip_pram = options['clip_ratio']
        self.entrop_coef = options['entrop_coef']
        self.value_coef = options['value_coef']
        self.lr = options['lr']
        self.clip_grad = options['clip_grad']

        self.optimizer.build(self.actor.get_trainable_values() + self.critic.get_trainable_values())
        self.optimizer.lr.assign(self.lr)


    def save(self, path):
        self.actor.save_weights(path+'_actor')
        self.critic.save_weights(path+'_critic')

    def load(self, path):
        self.actor.load_weights(path+'_actor')
        self.critic.load_weights(path+'_critic')
    
    def get_action_value(self,state, action=None):
        logits = self.actor(state)
        #std = tf.exp(self.actor.log_std)
        value = self.critic(state)
        # Discrete action space
        probs = tfp.distributions.Categorical(logits=logits)
        #print('Mean',mean,'Probs',probs)
        if action is None:
            action = probs.sample([1])[0]
        
        #print('Action',action.numpy()[0])

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        #print('Entropy',entropy)

        # Continuous action space
        #probs = tfp.distributions.Normal(mean, std)

        #if action is None:
        #   action = probs.sample([1])[0]
        
        #log_prob = tf.reduce_sum(probs.log_prob(action),axis=1)
        #entropy = tf.reduce_sum(probs.entropy(),axis=1)
        #print('Entropy',entropy,'Logprob',log_prob,'Value',value)

        return action, log_prob , entropy , value
   

    
    def learn(self,states,actions,adv,old_probs,old_value,returns):
        
        #print(adv.shape)
        #adv = tf.reshape(adv, (len(adv),))


        #old_p = tf.reshape(old_p, (len(old_p),2))
        with tf.GradientTape() as tape:
            _ , new_probs, new_entropy, new_value = self.get_action_value(states,actions)
            # Policy Loss
            logratio = new_probs - old_probs
            ratio = tf.exp(logratio)

            # Info values
            old_approx_kl = tf.reduce_mean(-logratio)
            approx_kl = tf.reduce_mean((ratio - 1) - logratio)
            clip_fracs = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1), self.clip_pram), tf.float32))
            #print('Advantage',adv.shape)
            #print('Ratio',ratio.shape)

            pg_loss1 = -adv * ratio
            #print('Ration Values',ratio)
            #print('Advantage',adv)
            #print('Clipparm',self.clip_pram)
            pg_loss2 = -adv * tf.clip_by_value(ratio, 1 - self.clip_pram, 1 + self.clip_pram)
            pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

            # Value Loss (Clip Value loss)
            v_loss_unclipped = tf.square(new_value - returns)
            v_clipped = old_value + tf.clip_by_value(new_value - old_value, -self.clip_pram, self.clip_pram)

            v_loss_clipped = tf.square(v_clipped - returns)
            v_loss_max = tf.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * tf.reduce_mean(v_loss_max)

            # Entropy Loss
            entropy_loss = tf.reduce_mean(new_entropy)
            #print('Entropy_loss',entropy_loss,'Entropy',new_entropy,'policy',pg_loss,'Value',v_loss)

            # Total Loss
            loss = pg_loss - self.entrop_coef * entropy_loss + self.value_coef * v_loss

        trainables = self.actor.get_trainable_values() + self.critic.get_trainable_values()

        grads = tape.gradient(loss, trainables)
        #for grad in grads:
        #print('grads',grads)
        grads, _ = tf.clip_by_global_norm(grads, self.clip_grad)
        #grads = [tf.clip_by_norm(grad, self.clip_grad) for grad in grads]

        self.optimizer.apply_gradients(zip(grads, trainables))

        return loss , v_loss, pg_loss, entropy_loss  , approx_kl , clip_fracs
    
    def set_lr(self,lr):
        self.lr = lr
        self.optimizer.lr.assign(lr)

    def anneal_lr(self,step,total_steps):
        frac = 1.0 - (step - 1.0) / total_steps
        
        real_lr = frac * self.lr
        self.optimizer.lr.assign(frac * self.lr)

        return real_lr


def gae(rewards,values,done,gamma=0.99,lmbda=0.95):
    #print(rewards.shape,values.shape,done.shape)
    #print(values)
    #print(rewards.shape,values.shape,done.shape)
    advantages = np.zeros_like(rewards)
    g = 0
    #print(rewards.shape,values.shape,done.shape)
    for i in reversed(range(rewards.shape[1])):
        non_terminal = 1 - int(done[i + 1])
        delta = rewards[:,i] + gamma * non_terminal * values[:,i + 1] - values[:,i]
        g = delta + gamma * lmbda  * non_terminal  * g
        
        advantages[:,i] = g
        
    returns = advantages + values[:,:-1]

    print('Avg_return',np.mean(returns,axis=1),'Avg_Adv' ,np.mean(advantages,axis=1))
    
    return returns,advantages 

def preprocess_data(states,actions,rewards,done,values,gamma=0.99,lmbda=0.95):

    returns, adv = gae(rewards,values,done,gamma,lmbda)
    #print(adv,returns)
    #adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
    #print(adv.shape)
    #states = np.array(states,dtype=np.float32)
    #actions = np.array(actions,dtype=np.float32)
    #returns = np.array(returns,dtype=np.float32)

    return states, actions, returns, adv




