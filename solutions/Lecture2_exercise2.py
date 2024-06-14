
import numpy as np
import tqdm
import matplotlib.pyplot as plt

def sim_uniform(n):
    
    sample = np.random.random(1)
        
    return (np.floor(sample * n).astype(np.int32) + 1).item()

def sim_geometric(p):
    
    U = np.random.random(1)
    
    return (np.floor(np.log(U) / np.log(1 - p)).astype(np.int32) + 1).item()


def sim_kpoint_direct(pks):
    
    p_cum = np.cumsum(pks)
    U = np.random.random(1)
    
    return np.sum(p_cum < U) + 1
    
    
def sim_kpoint_rejection(p):
    
    c = np.max(p)
    k = len(p)
    
    while True:
        I = sim_uniform(k)
        U2 = np.random.random(1)
        
        if U2 <= p[I-1] / c:
            return I # Accept I
        # else: Reject I
    
def get_alias_table(p):
    # Step 1: Initialization
    eps = 1e-6
    k = len(p)
    L = np.arange(k)
    
    # Step 2: Initial F tables
    F = k * p
    
    G = np.argwhere(F >= 1).squeeze()
    S = np.argwhere(F <= 1).squeeze()
    
    while len(S) > 0:
        i = G[0]
        j = S[0]
        
        L[j] = i
        F[i] = F[i] - (1- F[j])
        
        if F[i] < 1-eps:
            G = G[1:]
            S = np.append(S[1:], i)
        else:
            S = S[1:]
        
    return L, F
    
def sim_kpoint_alias(L, F):
    k = len(L)
    
    I = sim_uniform(k)
    U = np.random.random(1)
    
    if U <= F[I-1]:
        return I
    else:
        return L[I-1]
    

if __name__ == "__main__":
    
    n = 10_000
    
    geom = np.zeros(n)
    p6_crude = np.zeros(n)
    p6_rejection = np.zeros(n)
    p6_alias = np.zeros(n)
    
    p = np.array([7/48,5/48,1/8,1/16,1/4,5/16])
    # p = np.array([17/96,1/12,1/3,1/4,1/24,11/96])
    L, F = get_alias_table(p)
    
    for i in tqdm.tqdm(range(n)):
        geom[i] = sim_geometric(0.5)
        p6_crude[i] = sim_kpoint_direct(p)
        p6_rejection[i] = sim_kpoint_rejection(p)
        p6_alias[i] = sim_kpoint_alias(L, F)
        
    fig, axs = plt.subplots(2,4)
    
    axs[0,0].hist(geom, bins=10, density=True)
    axs[0,1].hist(np.random.geometric(0.5, n), bins=10, density=True)
    axs[1,0].hist(p6_crude, bins=6, density=True)
    axs[1,1].hist(p6_rejection, bins=6, density=True)
    axs[1,2].hist(p6_alias, bins=6, density=True)
    axs[1,3].hist(np.random.choice(np.arange(1,7), n, p=p), bins=6, density=True)
    
    plt.show()