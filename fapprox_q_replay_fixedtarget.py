# -*- coding: utf-8 -*-
"""

@author: ligraf

basic code from: https://keon.io/deep-q-learning/

target net from: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

"""

import random
import gym
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import matplotlib.pylab as plt  

# define directory
# path_to_file = "/Users/linus/Documents/OneDrive/Uni/Master/Reinforcment Learning/project"
path_to_file = r"C:\Users\tobia\OneDrive\Studium\Reinforcement Learning\project"
# path_to_file = r"./Documents/OneDrive/Uni/Master/Reinforcment Learning/project"


#%%
"""
Deep Q-learning Neural Network
"""
# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma = 0.95, epsilon_decay = 0.99, learning_rate = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = gamma    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        # initiate fixed target model
        self.target_model = self._build_model()
        self.tau = 0.05
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def model_summary(self):
        # Neural Net for Deep-Q learning Model
        return  self.model.summary()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
     
    def replay_new(self, batch_size):
        """
        Work in progress based on https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, new_state, done in minibatch:
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(self.target_model.predict(new_state)[0])
            self.model.fit(state, target, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        """
        Work in progress
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)   
        
    def play(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])        
            
    def load_weights(self, name):
        self.model.load_weights(name)
        print("Loaded estimated model")

    def save_weights(self, name):
        self.model.save_weights(name)
        print("Saved model to disk")
        
        
#%% FUNCTIONS

# define function to create state vector (dummy vector)            
def state_vec(state,state_size):
    vec = np.identity(state_size)[state:state + 1]
    return vec

# plot learning curve
from statsmodels.nonparametric.smoothers_lowess import lowess
def plot_learning_rate(dataset, ylabel="", xlabel=""):
    loess = pd.DataFrame(lowess(dataset.iloc[:,0], exog = dataset.iloc[:,1] ,frac=0.1, return_sorted=False))
    result = pd.concat([dataset,loess], axis=1) 
    result.columns = ['r', 'epochs','loess']
    # create plot
    plt.plot('epochs','r', data=result, color='gray')
    plt.plot('epochs','loess', data=result, color='red')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel) 

def play_game(env, agent, num_episodes, print_reward = False, text = True):
    """
    Play the game given in the env based on the policy in the given q- table.
    """
    total_epochs, reward_total = 0, 0
    n = env.observation_space.n
    
    for i in range(num_episodes):
        # print(i)
        state = env.reset()
        epochs, reward, cum_reward = 0, 0, 0
        done = False
        steps = 0
        while not done:
            action = agent.play(state_vec(state,n))
            state, reward, done, info = env.step(action)
            cum_reward += reward
            epochs += 1
            steps += 1

        total_epochs += epochs
        reward_total += cum_reward
        if print_reward == True:
            print(f"Epoch: {i} Steps: {epochs} Reward: {cum_reward}")
            
    if text == True:
        print("-----------------------------------------")
        print(f"Results after {num_episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / num_episodes}")
        print(f"Average reward per episode: {reward_total / num_episodes}")
    
    avr_timesteps = total_epochs / num_episodes
    avr_reward = reward_total / num_episodes
    
    return avr_reward, avr_timesteps      
        
#%%  
#"""
#Play CartPole
#"""
            
EPISODES = 10

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
# agent.load("./save/cartpole-dqn.h5")
done = False
batch_size = 32

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for episode in range(500):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, episode, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
#%%
"""
Play Taxi-v2
"""
game_name = "taxi_64_64_32_v2"

store = pd.HDFStore(path_to_file + "/saved_models/" + game_name +".h5")
# store['df']  # load it                   
            
episodes = 1001
gamma = 0.95 
epsilon_decay = 0.9999
learning_rate = 0.001

env = gym.make("Taxi-v2")
state_size = env.observation_space.n
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size,
                 gamma=gamma,
                 epsilon_decay=epsilon_decay,
                 learning_rate=learning_rate)
# load the saved weights
# agent.load("./save/cartpole-dqn.h5")
done = False
batch_size = 32
r_avg_list = []
r_list = []
start_time = time.time()
max_steps = 500

for episode in range(episodes):
    state = env.reset()
    state = state_vec(state,state_size)
    r_sum = 0
    for step in range(max_steps):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward # if not done else 100 # hack to not get stuck in a local minimum (agent could just always play 199 steps)
        next_state = state_vec(next_state,state_size)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        r_sum += reward
        if done:
            break
        if len(agent.memory) > batch_size:
            agent.replay_new(batch_size)
            agent.target_train()
    r_avg_list.append(r_sum / step)
    r_list.append(r_sum)
    print("episode: {}/{}, steps: {}, reward:{}, e: {:.2}"
                  .format(episode, episodes, step, r_sum, agent.epsilon))
    # safe the weights from time to time
    if episode % 100 == 0:
        agent.save_weights(path_to_file + "/saved_models/" + game_name +"_weights.h5")

run_time = round(((time.time() - start_time)/60),2)
avr_time = round(run_time/episodes,4)

# safe the learning progress
r_avg_list_table = pd.DataFrame(r_avg_list)
r_list_table = pd.DataFrame(r_list)
r_avg_list_table.to_hdf(store,'r_avg') 
r_list_table.to_hdf(store,'r') 
        
print("Time to run: {} min for {} episodes -- avr per episode: {} min".format(run_time, episodes, avr_time))   

plt.plot(r_avg_list)
plt.ylabel('Average reward per step')
plt.xlabel('Number of games')
plt.show()

plt.plot(r_list)
plt.ylabel('Reward per episode')
plt.xlabel('Number of games')
plt.show()

#%% Load the model and analyse its performance


game_name = "taxi_1000ep_48_24"

# first set up the agent
env = gym.make("Taxi-v2")
state_size = env.observation_space.n
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# load the model weights
agent.load_weights(path_to_file + "/saved_models/" + game_name +"_weights.h5")

# inspect keys in hdf store
with pd.HDFStore(path_to_file + "/saved_models/" + game_name +".h5") as hdf:
    print("HDF Keys:")
    print(hdf.keys())
    learning_return = hdf.get(key="r")
    
# load learning curve and plot it  
learning_return["epochs"] = learning_return.index.tolist()
learning_return.columns = ['r', 'epochs']
plot_learning_rate(learning_return,"Return","Iteration")

print(agent.model_summary())

# play game
num_episodes = 100
play_game(env, agent, num_episodes, print_reward = True, text = True)

