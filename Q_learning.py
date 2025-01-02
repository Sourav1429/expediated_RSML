# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:17:50 2025

@author: gangu
"""

import numpy as np
from Machine_Rep import Machine_Replacement
import pandas as pd
# Define the parameters
num_states = 4
num_actions = 2
gamma = 0.9  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration probability
num_episodes = 500  # Total episodes

# Initialize Q-table
Q = np.zeros((num_states, num_actions))

# Transition probability matrix (P[s, a, s_next])
mr_obj = Machine_Replacement()
P = mr_obj.gen_probability()

# Reward matrix (R[s, a])
R = mr_obj.gen_expected_reward()
pol_dict = {bin(i)[2:].zfill(4): i for i in range(16)}
T = num_episodes
df = {"run0":np.zeros(T),"run1":np.zeros(T),"run2":np.zeros(T),"run3":np.zeros(T)}
runs = len(df.keys())
# Simulate the environment for a given state and action
def step(state, action):
    probabilities = P[action,state]
    next_state = np.random.choice(range(num_states), p=probabilities)
    reward = R[action, state]
    return next_state, reward

# Q-learning algorithm
for run in range(runs):
    Q = np.zeros((num_states, num_actions))
    for episode in range(num_episodes):
        state = np.random.randint(0, num_states)  # Start at a random state
        done = False
    
        while not done:
            # Choose action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.randint(0, num_actions)  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit
    
            # Take action and observe next state and reward
            next_state, reward = step(state, action)
    
            # Q-learning update
            best_next_action = np.argmax(Q[next_state])
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * Q[next_state, best_next_action] - Q[state, action]
            )
    
            # Update state
            state = next_state
    
            # Check if terminal state is reached
            #optimal_policy = np.argmax(Q, axis=1)
            policy = ""
            for j in range(num_states):
                policy = policy + str(np.argmax(Q[j]))
            #print(policy)
            df["run"+str(run)][episode] = pol_dict[policy]
            if state == num_states - 1:  # Assuming last state is terminal
                done = True

# Derive the optimal policy from Q-table
dF = pd.DataFrame(df)
dF.to_csv("Policy_selected_Qlearning.csv")