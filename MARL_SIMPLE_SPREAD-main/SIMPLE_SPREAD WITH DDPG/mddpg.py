import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
import gym
import pettingzoo
from pettingzoo.mpe import simple_spread_v1
env =simple_spread_v1.parallel_env(max_frames=100)
from tensorflow.keras.models import load_model
model1=load_model('AGENT_0.h5')
model2=load_model('AGENT_1.h5')
model3=load_model('AGENT_2.h5')
actor_models={'agent_0':model1,'agent_1':model2,'agent_2':model3}

upper_bound=1
lower_bound=-1
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
ep_reward_list=[]
episode_steps1=[] 
def policy(agent,state,noise_object):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sampled_actions = tf.squeeze(actor_models[agent](state))
        noise = noise_object()
        sampled_actions = sampled_actions.numpy() + noise
        sampled_actions=np.clip(sampled_actions,lower_bound,upper_bound)
        return sampled_actions
    
for episode in range(5000):

    last_states = env.reset()
    sum_rewards = {'agent_0':0,'agent_1':0,'agent_2':0}
    episode_steps=0
    while True:
   
        actions = {agent: policy(agent,last_states[agent],ou_noise) for agent in env.agents}
        next_states,rewards,done,info=env.step(actions)
        env.render()
             
        sum_rewards={agent: (sum_rewards[agent]+rewards[agent]) for agent in rewards}
        episode_steps+=1
        
        if done['agent_0']:
            break
        last_states=next_states
    
        
    ep_reward_list.append(sum_rewards)
    episode_steps1.append(episode_steps)
        
