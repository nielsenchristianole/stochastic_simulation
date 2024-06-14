
import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm

def metropolis_hasting(dist, proposal,initial, n_iter = 10_000):
    
    chain = [initial]
    samples = []
    
    for i in tqdm.trange(n_iter):
        
        # Discrete uniform proposal
        sample = proposal(chain[-1])
        A = min(1, dist(sample)/dist(chain[-1]))
        
        if np.random.rand() < A:
            chain.append(sample)
        else:
            chain.append(chain[-1])
            
    return np.array(chain)

def metropolis_coordinate_wise(dist, prop_x, prop_y ,initial, n_iter = 10_000):
        
    chain = [initial]
    samples = []
    
    for i in tqdm.trange(n_iter):
        
        # Discrete uniform proposal
        if i % 2 == 0:
            sample = [prop_x(chain[-1][0]), chain[-1][1]]
        else:
            sample = [chain[-1][0], prop_y(chain[-1][1])]
            
        A = min(1, dist(sample)/dist(chain[-1]))
        
        if np.random.rand() < A:
            chain.append(sample)
        else:
            chain.append(chain[-1])
            
    return np.array(chain)

def gibbs_sampler(dist, cond1, cond2, initial, n_iter = 10_000):
        
    chain = [initial]
    
    for i in tqdm.trange(n_iter):
        
        sample = [cond1(chain[-1][1]), cond2(chain[-1][0])]
        chain.append(sample)
        
    return np.array(chain)

def ex1():
    chain_length = 100_000
    
    x_0 = 1
    A = 8
    dist = lambda x : A**x / math.factorial(x)
    
    def proposal(x):
        return max(np.uint32(0), x + np.random.choice([-3,-2,-1, 1,2,3])).astype(np.uint32)
    
    chain = metropolis_hasting(dist, proposal, x_0, n_iter = chain_length)
    
    true_samples = np.random.poisson(A, len(chain))
    
    bin_count = 20
    bins = np.linspace(0, np.max([np.max(chain), np.max(true_samples)]), bin_count)
    f_obs = np.histogram(chain, bins = bins, density=True)[0]
    f_exp = np.histogram(true_samples, bins = bins, density=True)[0]
    p = 1-stats.chisquare(f_obs, f_exp)[1]
    
    fig, axs = plt.subplots(1,2)
    axs[0].hist(chain, bins = bins)
    axs[1].hist(true_samples, bins = bins)
    plt.show()
    
    print(f"Chi-square test p-value: {p}")
    
def ex2_1():
    chain_length = 100_000
    
    x0 = [1,1]
    A1, A2 = 4, 4
    dist = lambda x : (A1**x[0] * A2**x[1]) / (math.factorial(x[0]) * math.factorial(x[1]))
    
    def proposal(x):
        coord1 = max(np.uint32(0),x[0] + np.random.choice([-3,-2,-1, 1,2,3])).astype(np.uint32)
        coord2 = max(np.uint32(0),x[1] + np.random.choice([-3,-2,-1, 1,2,3])).astype(np.uint32)
        return [coord1, coord2]
    
    chain = metropolis_hasting(dist, proposal, x0, n_iter = chain_length)
    
    true1 = np.random.poisson(A1, len(chain))
    true2 = np.random.poisson(A2, len(chain))
    
    bin_count = 20
    bins1 = np.linspace(0, np.max([np.max(chain[:,0]), np.max(true1)])+1, bin_count)
    bins2 = np.linspace(0, np.max([np.max(chain[:,1]), np.max(true2)])+1, bin_count)
    
    f_obs = np.histogram2d(chain[:,0], chain[:,1], bins = [bins1, bins2], density=True)[0]
    f_exp = np.histogram2d(true1, true2, bins = [bins1, bins2], density=True)[0]
    
    # plot the two histograms
    fig, axs = plt.subplots(1,2)
    axs[0].hist2d(chain[:,0], chain[:,1], bins = [bins1, bins2])
    axs[1].hist2d(true1, true2, bins = [bins1, bins2])
    plt.show()

def ex2_2():
    chain_length = 100_000
    
    x0 = [1,1]
    A1, A2 = 4, 4
    dist = lambda x : (A1**x[0] * A2**x[1]) / (math.factorial(x[0]) * math.factorial(x[1]))
    
    def proposal(x):
        return max(np.uint32(0), x + np.random.choice([-3,-2,-1, 1,2,3])).astype(np.uint32)
    
    chain = metropolis_coordinate_wise(dist, proposal, proposal, x0, n_iter = chain_length)
    
    true1 = np.random.poisson(A1, len(chain))
    true2 = np.random.poisson(A2, len(chain))
    
    bin_count = 20
    bins1 = np.linspace(0, np.max([np.max(chain[:,0]), np.max(true1)])+1, bin_count)
    bins2 = np.linspace(0, np.max([np.max(chain[:,1]), np.max(true2)])+1, bin_count)
    
    f_obs = np.histogram2d(chain[:,0], chain[:,1], bins = [bins1, bins2], density=True)[0]
    f_exp = np.histogram2d(true1, true2, bins = [bins1, bins2], density=True)[0]
    
    # plot the two histograms
    fig, axs = plt.subplots(1,2)
    axs[0].hist2d(chain[:,0], chain[:,1], bins = [bins1, bins2])
    axs[1].hist2d(true1, true2, bins = [bins1, bins2])
    plt.show()
    
def ex2_3():
    chain_length = 100_000
    
    x0 = [1,1]
    A1, A2 = 4, 4
    dist = lambda x : (A1**x[0] * A2**x[1]) / (math.factorial(x[0]) * math.factorial(x[1]))
    
    def cond1(x):
        return np.random.poisson(A1)
    def cond2(x):
        return np.random.poisson(A2)
    
    chain = gibbs_sampler(dist, cond1, cond2, x0, n_iter = chain_length)
    
    true1 = np.random.poisson(A1, len(chain))
    true2 = np.random.poisson(A2, len(chain))
    
    bin_count = 20
    bins1 = np.linspace(0, np.max([np.max(chain[:,0]), np.max(true1)])+1, bin_count)
    bins2 = np.linspace(0, np.max([np.max(chain[:,1]), np.max(true2)])+1, bin_count)
    
    f_obs = np.histogram2d(chain[:,0], chain[:,1], bins = [bins1, bins2], density=True)[0]
    f_exp = np.histogram2d(true1, true2, bins = [bins1, bins2], density=True)[0]
    
    # plot the two histograms
    fig, axs = plt.subplots(1,2)
    axs[0].hist2d(chain[:,0], chain[:,1], bins = [bins1, bins2])
    axs[1].hist2d(true1, true2, bins = [bins1, bins2])
    plt.show()
    
def ex3():
    
    # (a), generate pairs of theta and phi
    n_a = 10
    gamma = np.random.normal(0, 1)
    xi = np.random.normal(0, 1)
    theta = np.exp(gamma)
    phi = np.exp(gamma)
    
    # (b) generate X_i
    X_i = np.random.normal(theta, phi, n_a)
    
    # (c) derive the posterior distribution
    # (d) generate MCMC samples
    # (e) repeat hte experiment

if __name__ == "__main__":
    
    # ex1()
    # ex2_1()
    # ex2_2()
    # ex2_3()
    ex3()
    
    #ex2
    
