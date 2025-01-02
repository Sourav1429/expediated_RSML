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
T = 500
beta = 0.1/(np.arange(1,T+1))
nS,nA = mr_obj.nS,mr_obj.nA
N_max = 100
mu_sa = R
sigma = 0
gamma = 0.95
ch =2
df = {"run0":np.zeros(T),"run1":np.zeros(T),"run2":np.zeros(T),"run3":np.zeros(T)}
runs = len(df.keys())
vf_obj = get_min_vf(psi, beta, T, nS, nA, N_max, mu_sa, sigma, gamma,P,ch)
print(vf_obj.get_Q_table())
for run in range(runs):
    vf_obj = get_min_vf(psi, beta, T, nS, nA, N_max, mu_sa, sigma, gamma,P,ch)
    vf_obj.get_vf("run"+str(run),df)
dF = pd.DataFrame(df)
dF.to_csv("Policy_selected_non_robust1.csv")
print(vf_obj.get_Q_table())
#Storing the Q-table

q_tab = vf_obj.get_Q_table()
import pickle
with open("Non_robust_MR_chk_q_tab","wb") as f:
    pickle.dump(q_tab,f)
f.close()



