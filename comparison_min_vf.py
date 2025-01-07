# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 08:32:34 2025

@author: Sourav
"""

import numpy as np

from Machine_Rep import Machine_Replacement
from finding_min_vf import get_min_vf
import pandas as pd

class min_value_func:
    def __init__(self,P_set,n_actions,actions,init_state,gamma):
        self.P_set = P_set
        self.n_actions = n_actions
        self.actions = actions
        self.init_state = init_state
        self.gamma = gamma
    def find_vf(self,preparedP,R,policy):
        I = np.eye(self.n_actions)
        print(I)
        Q = self.gamma*np.dot(policy[self.init_state],np.transpose(preparedP))
        print(Q)
        return np.dot(np.linalg.pinv(I-Q),np.dot(policy[self.init_state,:],R))
    def obtain_robust_vf(self,policy,R):
        state = self.init_state
        vf = []
        for P in self.P_set:
            preparedP = []
            for a in self.actions:
                preparedP.append(P[a,state,:])
            preparedP = np.array(preparedP)
            V = self.find_vf(preparedP,R,policy)
            vf.append(V)
        pos = np.argmin(vf)
        return (vf[pos],pos)

mr_obj = Machine_Replacement()
R1,R,P,C = mr_obj.gen_expected_reward(ch=2),mr_obj.gen_reward(ch=2),mr_obj.gen_probability(),mr_obj.gen_expected_cost()
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
min_vf = min_value_func(choice_of_P, len(actions), actions, init_state, gamma)
policy = np.array([[1,0],[0,1],[0,1],[0,1]])
print(min_vf.obtain_robust_vf(policy, R1))
vf_obj = get_min_vf(psi, beta, T, nS, nA, N_max, mu_sa, sigma, gamma,P,ch)
print(np.average(vf_obj.get_Q_table()[init_state,:]))

