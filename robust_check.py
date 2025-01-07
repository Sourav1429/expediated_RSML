# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 14:46:10 2025

@author: gangu
"""
import numpy as np
from Machine_Rep import Machine_Replacement
from finding_min_vf import get_min_vf
import pandas as pd

mr_obj = Machine_Replacement()
R,P,C = mr_obj.gen_reward(ch=2),mr_obj.gen_probability(),mr_obj.gen_expected_cost()
psi = 0.5
T = 1000
beta = 0.1/(np.arange(1,T+1))
nS,nA = mr_obj.nS,mr_obj.nA
N_max = 100
mu_sa = R
sigma = 0
gamma = 0.95
ch =2
df = {"run0":np.zeros(T),"run1":np.zeros(T),"run2":np.zeros(T),"run3":np.zeros(T)}
runs = 1#len(df.keys())
#vf_obj = get_min_vf(psi, beta, T, nS, nA, N_max, mu_sa, sigma, gamma,P,ch)
#print(vf_obj.get_Q_table())
P1 = np.copy(P)
P1[0,1] = [0,0,0,1]

P2 = np.copy(P)
P2[0,1] = [0,0,0.01,0.99]

P3 = np.copy(P)
P3[0,1] = [0,0.33,0.11,1-(0.33+0.11)]

choice_of_P = np.array([P,P1,P2,P3])
'''
for run in range(runs):
    vf_obj = get_min_vf(psi, beta, T, nS, nA, N_max, mu_sa, sigma, gamma,P,choice_of_P,ch)
    vf_obj.get_vf("run"+str(run),df)
dF = pd.DataFrame(df)
dF.to_csv("Policy_selected_non_robust1.csv")
print(vf_obj.get_Q_table())
#Storing the Q-table

q_tab = vf_obj.get_Q_table()
import pickle
with open("Non_robust_MR_chk_q_tab_one_prob_uncertained","wb") as f:
    pickle.dump(q_tab,f)
f.close()'''

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
import pickle
with open("Q_table_or","wb") as f:
    pickle.dump(Q)
f.close()



