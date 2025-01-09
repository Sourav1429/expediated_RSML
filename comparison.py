# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:46:10 2025

@author: Sourav_
"""

import numpy as np

from Machine_Rep import Machine_Replacement
from finding_min_vf import get_min_vf
import pandas as pd
from Value_Iteration import ValueIteration


mr_obj = Machine_Replacement()
R1,R,P,C = mr_obj.gen_expected_reward(ch=2),mr_obj.gen_reward(ch=2),mr_obj.gen_probability(),mr_obj.gen_expected_cost()
print(R1)
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

init_state = 0
gamma = 0.95
actions = np.array([0,1])
#min_vf = min_value_func(choice_of_P, len(actions), actions, init_state, gamma)
policy = np.array([[1,0],[0,1],[0,1],[0,1]])

for p in choice_of_P:
    print("======================================")
    vi = ValueIteration(nS, nA, p, R1)
    vi.value_iteration()

    # Retrieve the value function and optimal policy
    value_function = vi.get_value_function()
    optimal_policy = vi.get_policy()
    
    print("Optimal Value Function:")
    print(value_function)
    
    print("Optimal Policy:")
    print(optimal_policy)
#print(find_min_vf_cf(policy, choice_of_P, R1, init_state))
#print(min_vf.obtain_robust_vf(policy, R1))
vf_obj = get_min_vf(psi,beta,T,nS,nA,N_max,mu_sa,sigma,gamma,P,choice_of_P,ch = 2)
vf_obj.get_vf("run"+str(0),df)
print("T-MLMC",np.average(vf_obj.get_Q_table()[init_state,:]))