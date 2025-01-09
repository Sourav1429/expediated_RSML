# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:43:59 2025

@author: gangu
"""

import numpy as np
from itertools import product
from Machine_Rep import Machine_Replacement,River_swim
from scipy.optimize import linprog

def find_feasible_distributions(x, sigma, num_samples=1000):
    """
    Finds feasible probability distributions v satisfying ||v - x||_1 <= sigma.

    Parameters:
    - x: array-like, the original probability distribution (vector).
    - sigma: float, the L1 constraint on the distance between v and x.
    - num_samples: int, number of random samples to generate for testing feasibility.

    Returns:
    - feasible_v: list of feasible probability distributions.
    """
    n = len(x)
    feasible_v = []

    for _ in range(num_samples):
        # Generate a random perturbation within [-sigma, sigma] for each element
        delta = np.random.uniform(-sigma / 2, sigma / 2, size=n)
        v_candidate = x + delta

        # Ensure the candidate is a probability distribution
        v_candidate = np.maximum(v_candidate, 0)  # No negative probabilities
        v_candidate /= np.sum(v_candidate)  # Normalize to sum to 1

        # Check if the L1 distance constraint is satisfied
        if np.sum(np.abs(v_candidate - x)) <= sigma:
            feasible_v.append(v_candidate)

    return feasible_v

# Example usage
env_ch = 0  # 0 is for Machine Replacement and 1 is for Riverswim
env_nm = ["MR","RS"]


if(env_ch==0):
    env_obj = Machine_Replacement()
elif(env_ch==1):
    env_obj = River_swim()
else:
    print("Invalid choice")
    
P0 = env_obj.gen_probability()
nS,nA = env_obj.nS,env_obj.nA
all_pairs = dict()
sigma = 0.01
step = sigma/10

for s in range(nS):
    for a in range(nA):
        p = P0[a,s]
        valid_vectors = find_feasible(p, sigma,step)
        all_pairs[(s,a)] = valid_vectors

print("Got all the valid pairs")

print("Combining to form uncertainity sets")

import pickle
with open("pairs_obtained_"+env_nm[env_ch],"wb") as f:
    pickle.dump(all_pairs,f)
f.close()
        
        

'''x = np.array([1.0, 2.0, 3.0, 4.0])  # Example vector x in R^4
sigma = 2.0  # L1 distance threshold
step_size = 0.5  # Sampling step size

result = generate_l1_ball(x, sigma, step_size)

# Display the result
print(f"Number of valid vectors: {len(result)}")
for vector in result:
    print(vector)'''
