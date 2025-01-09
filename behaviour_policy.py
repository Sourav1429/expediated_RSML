# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:43:23 2025

@author: gangu
"""
import numpy as np
class beh_pol_sd:
    def __init__(self,P,policy,nS,nA):
        self.P = P
        self.policy = policy
        self.nS = nS;
        self.nA = nA;
    
    def onehot(self):
        pol = np.zeros((self.nS,self.nA));
        for i in range(self.nS):
            pol[i][int(self.policy[i])]=1;
        return pol;
    def find_transition_matrix(self,onehot_encode=1):
        if(onehot_encode==1):
            self.policy = self.onehot()
        T_s_s_next = np.zeros((self.nS,self.nS));
        for s in range(self.nS):
            for s_next in range(self.nS):
                for a in range(self.nA):
                    #print(s,s_next,a);
                    #print(T[a,s,s_next]);
                    T_s_s_next[s,s_next]+=self.P[a,s,s_next]*self.policy[s,a];
        return T_s_s_next;
    def state_distribution_simulated(self,onehot_encode=1):
        P_policy = self.find_transition_matrix(onehot_encode)
        #print(P_policy);
        P_dash = np.append(P_policy - np.eye(self.nS),np.ones((self.nS,1)),axis=1);
        #print(P_dash);
        P_last = np.linalg.pinv(np.transpose(P_dash))[:,-1]
        return P_last;