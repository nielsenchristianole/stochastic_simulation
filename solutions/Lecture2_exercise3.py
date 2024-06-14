

import numpy as np
import scipy.stats as stats

def uniform(a,b):
    
    return a + (b-a) * np.random.random(1)

def exponential(lam):
    
    U = np.random.random(1)
    
    return -np.log(U) / lam

def pareto(k, beta):
    
    U = np.random.random(1)
    
    return beta*(U**(-1/k))

def gaussian(mu, sigma):
    
    U = np.random.random(1)
    
    return mu + sigma * np.sqrt(2) * stats.norm.ppf(U)
