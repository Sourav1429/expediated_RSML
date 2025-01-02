# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 18:37:03 2024

@author: Sourav
"""
import numpy as np
from scipy.optimize import minimize
import math

class get_min_vf:
    def __init__(self,psi,beta,T,nS,nA,N_max,mu_sa,sigma,gamma,P,ch = 2): #ch = 0,1 and 2 where 0 is TV, 1 is chi-square and 2 is for KL divergence
        self.psi = psi
        self.beta = beta
        self.T = T
        self.nS = nS
        self.nA = nA
        self.Q = np.zeros((nS,nA))
        self.N_max = N_max
        self.mu_sa = mu_sa
        self.gamma = gamma
        self.sigma = sigma
        self.ch = ch
        self.p_sa = P
        
    def robust_bellman(self,r_hat,V_rho):
        ret_val = r_hat + self.gamma*V_rho
        if(math.isnan(ret_val)):
            print("R_hat=",r_hat)
            print("V_rho=",V_rho)
            input();
        return ret_val
    
    def objective_KL(self,x,V_samples):
        if x <= 0:  # Ensure x > 0 to avoid division by zero or negative log
            return -np.inf
        eps = 0.0001
        expectation = np.mean(np.exp(-V_samples / x))+eps
        if(expectation ==0):
            print("x =",x)
        #print(expectation)
        return -(-x * np.log(expectation) - x * self.sigma)
    def objective_TV(self,x,samples):
        m_samples = np.clip(samples,a_max=x)
        min_val = np.min(m_samples)
        expectation = np.mean(m_samples) - self.sigma/2*(x-min_val)
        return -expectation
    def objective_chi_sq(self,x,samples):
        m_samples = np.clip(samples,a_max=x)
        ret_val = np.mean(m_samples) - np.sqrt(self.sigma*np.var(m_samples))
        return -ret_val
    def del_V_s_a(self,V_samples):
        if(self.ch ==2):
            lower_bound = 1e-3  # Prevent x from being too small
            constraints = [{'type': 'ineq', 'fun': lambda x: x[0] - lower_bound}]
            x0 = 1.0
            result = minimize(self.objective_KL, x0,args=(V_samples), constraints=constraints, bounds=[(lower_bound, None)])
            self.alpha1 = result.x[0]
            result = minimize(self.objective_KL, x0,args=(V_samples[::2]), constraints=constraints, bounds=[(lower_bound, None)])
            self.alpha2 = result.x[0]
            result = minimize(self.objective_KL, x0, args=(V_samples[1::2]),constraints=constraints, bounds=[(lower_bound,None)])
            self.alpha3 = result.x[0]
            vf1 = -self.alpha1*np.log(np.average(np.exp(-V_samples/self.alpha1))) - self.alpha1*self.sigma
            vf2 = -self.alpha2*np.log(np.average(np.exp(-V_samples[::2]/self.alpha2))) - self.alpha2*self.sigma
            vf3 = -self.alpha3*np.log(np.average(np.exp(-V_samples[1::2]/self.alpha3))) - self.alpha3*self.sigma
        elif(self.ch==0):
            constraints = [{'type': 'ineq', 'fun': lambda x: x}]
            x0 = 1.0
            result = minimize(self.objective_TV, x0,args=(V_samples), constraints=constraints, bounds=[(0, None)])
            vf1 = -result.fun
            result = minimize(self.objective_TV, x0,args=(V_samples[::2]), constraints=constraints, bounds=[(0, None)])
            vf2 = -result.fun
            result = minimize(self.objective_TV, x0, args=(V_samples[1::2]),constraints=constraints, bounds=[(0, None)])
            vf3 = -result.fun
        elif(self.ch==1):
            constraints = [{'type': 'ineq', 'fun': lambda x: x}]
            x0 = 1.0
            result = minimize(self.objective_chi_sq, x0,args=(V_samples), constraints=constraints, bounds=[(0, None)])
            vf1 = -result.fun
            result = minimize(self.objective_chi_sq, x0,args=(V_samples[::2]), constraints=constraints, bounds=[(0, None)])
            vf2 = -result.fun
            result = minimize(self.objective_chi_sq, x0, args=(V_samples[1::2]),constraints=constraints, bounds=[(0, None)])
            vf3 = -result.fun
        vf = vf1-vf2/2-vf3/2;
        if(math.isnan(vf)):
            print("VF1=",vf1)
            print("VF2=",vf2)
            print("VF3=",vf3)
        return vf
    def del_s_a(self,rew_samples):
        rew1 = 0;
        rew2 = 0;
        rew3 = 0;
        if self.ch==2:
            lower_bound = 1e-5  # Prevent x from being too small
            constraints = [{'type': 'ineq', 'fun': lambda x: x[0] - lower_bound}]
            x0 = 1.0
            result = minimize(self.objective_KL, x0,args=(rew_samples), constraints=constraints, bounds=[(lower_bound, None)])
            self.alpha1 = result.x[0]
            result = minimize(self.objective_KL, x0,args=(rew_samples[::2]), constraints=constraints, bounds=[(lower_bound, None)])
            self.alpha2 = result.x[0]
            result = minimize(self.objective_KL, x0, args=(rew_samples[1::2]),constraints=constraints, bounds=[(lower_bound,None)])
            self.alpha3 = result.x[0]
            rew1 = -self.alpha1*np.log(np.average(np.exp(-rew_samples/self.alpha1))) - self.alpha1*self.sigma
            rew2 = -self.alpha2*np.log(np.average(np.exp(-rew_samples[::2]/self.alpha2))) - self.alpha2*self.sigma
            rew3 = -self.alpha3*np.log(np.average(np.exp(-rew_samples[1::2]/self.alpha3))) - self.alpha3*self.sigma
        elif(self.ch==0):
            constraints = [{'type': 'ineq', 'fun': lambda x: x}]
            x0 = 1.0
            result = minimize(self.objective_TV, x0,args=(rew_samples), constraints=constraints, bounds=[(0, None)])
            rew1 = -result.fun
            result = minimize(self.objective_TV, x0,args=(rew_samples[::2]), constraints=constraints, bounds=[(0, None)])
            rew2 = -result.fun
            result = minimize(self.objective_TV, x0, args=(rew_samples[1::2]),constraints=constraints, bounds=[(0, None)])
            rew3 = -result.fun
        elif(self.ch==1):
            constraints = [{'type': 'ineq', 'fun': lambda x: x}]
            x0 = 1.0
            result = minimize(self.objective_chi_sq, x0,args=(rew_samples), constraints=constraints, bounds=[(0, None)])
            rew1 = -result.fun
            result = minimize(self.objective_chi_sq, x0,args=(rew_samples[::2]), constraints=constraints, bounds=[(0, None)])
            rew2 = -result.fun
            result = minimize(self.objective_chi_sq, x0, args=(rew_samples[1::2]),constraints=constraints, bounds=[(0, None)])
            rew3 = -result.fun
        rew = rew1 - rew2/2 - rew3/2 
        if(math.isnan(rew)):
            print("Rew1=",rew1)
            print("Rew2=",rew2)
            print("Rew3=",rew3)
            input()
        return rew
    def get_reward_integrated(self,rew_samples,N1):
        P_N1 = self.psi*np.power((1-self.psi),N1)
        rew = rew_samples[0]+(self.del_s_a(rew_samples)/P_N1)
        return rew
    def get_integrated_vf(self,V,s,a,N2):
        P_N2 = self.psi*np.power((1-self.psi),N2)
        del_V_sa_ = self.del_V_s_a(V)
        vf = V[0]+del_V_sa_/P_N2
        if(math.isnan(vf)):
            print("V_0=",V[0])
            print("delV_sa=",del_V_sa_)
            print("P_N2=",P_N2)
            input()
        return vf
    def get_vf(self,run,df):
        V,pol = np.zeros(self.nS),np.zeros(self.nS)
        pol_dict = {bin(i)[2:].zfill(4): i for i in range(16)}
        for t in range(self.T):
            for s in range(self.nS):
                V[s] = np.max(self.Q[s,:])
                pol[s] = np.argmax(self.Q[s,:])
            for s in range(self.nS):
                for a in range(self.nA):
                    N1,N2 = np.random.geometric(self.psi),np.random.geometric(self.psi)
                    NN1 = 1 + np.power(2,N1+1)*int((N1 <= self.N_max))
                    NN2 = 1 + np.power(2,N2+1)*int((N2 <= self.N_max))
                    rew_samples = self.mu_sa[a,s,np.random.choice(len(self.mu_sa[0,0]),NN1)]##careful here
                    #Computing \hat{r^{\rho(\sigma)}}(s,a)
                    r_hat = self.get_reward_integrated(rew_samples,N1)
                    #print("r_hat",r_hat)
                    next_state_samples = np.random.choice(len(self.p_sa[0,0]),NN2,p=self.p_sa[a,s])
                    #self.p_sa[s,a,np.random.choice(len(self.p_sa[0,0]),NN2)]#careful here
                    #Updating Q values
                    V_samples = V[next_state_samples]
                    V_rho = self.get_integrated_vf(V_samples,s,a,N2)
                    #print("V_rho=",V_rho)
                    self.Q[s,a] = (1-self.beta[t])*self.Q[s,a] + self.beta[t]*self.robust_bellman(r_hat,V_rho)
            policy = ""
            for j in range(self.nS):
                policy = policy + str(np.argmax(self.Q[j]))
            df[run][t] = pol_dict[policy]
        print("Q-table updation complete")
    def get_Q_table(self):
        return self.Q
        
                
                
        
