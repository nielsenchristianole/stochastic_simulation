
import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm

def metropolis_hasting(dist, proposal,initial, n_iter = 10_000):
    
    chain = [initial]
    samples = []
    
    for i in range(n_iter):
        
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
    
    for i in range(n_iter):
        
        # Discrete uniform proposal
        if i % 2 == 0:
            sample = [prop_x(chain[-1][0], chain[-1][1]), chain[-1][1]]
        else:
            sample = [chain[-1][0], prop_y(chain[-1][1],chain[-1][0])]
            
        A = min(1, dist(sample)/dist(chain[-1]))
        
        if np.random.rand() < A:
            chain.append(sample)
        else:
            chain.append(chain[-1])
            
    return np.array(chain)

def gibbs_sampler(dist, cond1, cond2, initial, n_iter = 10_000):
        
    chain = [initial]
    
    for i in range(n_iter):
        
        if i % 2 == 0:
            sample = [cond1(chain[-1][1]), chain[-1][1]]
        else:
            sample = [chain[-1][0], cond2(chain[-1][0])]
            
        chain.append(sample)
        
    return np.array(chain)

def get_multivariate_dist(A1, A2, m):
    tmps1 = [[A1**x*A2**y/(math.factorial(x)*math.factorial(y))for x in range(m+1-y)] for y in range(m+1)]
    ps = np.zeros((m+1,m+1))
    for i, tmp in enumerate(tmps1):
        ps[:m+1-i,i] = np.array(tmp)
    
    ps /= np.sum(ps)
    
    return ps

def ex1():
    chain_length = 50_000
    
    x_0 = 1
    A = 8
    m = np.uint32(10)
    dist = lambda x : A**x / math.factorial(x)
    
    def proposal(x):
        val = x + np.random.choice([-3,-2,-1, 1,2,3])
        return max(min(val, m-1), np.uint32(0)).astype(np.uint32)
        # return max(np.uint32(0), x + np.random.choice([-3,-2,-1, 1,2,3])).astype(np.uint32)
    
    chain = metropolis_hasting(dist, proposal, x_0, n_iter = chain_length)
    
    ps = [A**x/math.factorial(x) for x in range(m+1)]
    true_samples = np.random.choice(range(m+1), p = ps/np.sum(ps), size = chain_length)
    
    bins = np.arange(m+1)
    f_obs = np.histogram(chain, bins = bins, density=True)[0]
    f_exp = np.histogram(true_samples, bins = bins, density=True)[0]
    p = stats.chisquare(f_obs, f_exp)[1]
    
    fig, axs = plt.subplots(1,2)
    axs[0].hist(chain, bins = bins)
    axs[1].hist(true_samples, bins = bins)
    
    fig.tight_layout()
    fig.suptitle("Hasting 1d")
    fig.savefig("methasting1d.png")
    plt.show()
    
    print(f"(Hasting 1d) Chi-square test p-value: {p}")
    
def extract_lower_triangle(matrix):
    n = matrix.shape[0]
    indices = np.tril_indices(n)
    return np.rot90(matrix)[indices]
    
def ex2_1():
    chain_length = 100_000
    
    x0 = [1,1]
    A1, A2 = 4, 4
    m = np.uint32(10)
    dist = lambda x : (A1**x[0] * A2**x[1]) / (math.factorial(x[0]) * math.factorial(x[1]))

    def proposal(x):
        
        val1 = np.random.randint(0, m+1)
        val2 = np.random.randint(0, m+1-val1)

        return [val1, val2]
    
    chain = metropolis_hasting(dist, proposal, x0, n_iter = chain_length)
    
    ps = get_multivariate_dist(A1, A2, m)
    
    obs = np.zeros((m+1,m+1))
    for element in chain:
        obs[element[0], element[1]] += 1
    obs /= np.sum(obs)
    
    # plot the two histograms
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(obs, origin = 'lower', extent = (0,m,0,m))
    axs[0].set_title("Observed")
    axs[0].set_aspect('equal')
    axs[1].imshow(ps, origin = 'lower', extent = (0,m,0,m))
    axs[1].set_title("Expected")
    axs[1].set_aspect('equal')
    
    fig.tight_layout()
    
    # set title
    fig.suptitle("Hasting 2d")
    fig.savefig("methasting2d.png")
    # plt.imsave("methasting2d.png", np.hstack([obs, ps]), cmap = 'viridis')
    plt.show()
    
    ps = extract_lower_triangle(ps)
    obs = extract_lower_triangle(obs)
    
    p = stats.chisquare(obs, ps)[1]
    
    print(f"(Hasting 2d) Chi-square test p-value: {p}")

def ex2_2():
    chain_length = 100_000
    
    x0 = [1,1]
    A1, A2 = 4, 4
    m = 10
    dist = lambda x : (A1**x[0] * A2**x[1]) / (math.factorial(x[0]) * math.factorial(x[1]))
    
    def proposal(x1, x2):
        x1 = x1 + np.array([-3,-2, -1, 0, 1,2,3])
        x1 = x1[x1 >= 0]
        x1 = x1[x1 < m+1-x2]
        x1 = np.random.choice(x1)
        return x1
    
    chain = metropolis_coordinate_wise(dist, proposal, proposal, x0, n_iter = chain_length)
    
    ps = get_multivariate_dist(A1, A2, m)
    
    obs = np.zeros((m+1,m+1))
    for element in chain:
        obs[element[0], element[1]] += 1
    obs /= np.sum(obs)
    
    # plot the two histograms
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(obs, origin = 'lower')
    axs[0].set_title("Observed")
    axs[0].set_aspect('equal')
    axs[1].imshow(ps, origin = 'lower')
    axs[1].set_title("Expected")
    axs[1].set_aspect('equal')
    
    fig.tight_layout()
    fig.suptitle("Coordinate-wise Metropolis-Hasting")
    fig.savefig("metropolis2d_coordwise.png")
    
    plt.show()
    
    ps = extract_lower_triangle(ps)
    obs = extract_lower_triangle(obs)
    
    p = stats.chisquare(obs, ps)[1]
    
    print(f"(Coord-wise) Chi-square test p-value: {p}")
    
def ex2_3():
    chain_length = 100_000
    
    x0 = [1,1]
    m = 10
    A1, A2 = 4, 4
    dist = lambda x : (A1**x[0] * A2**x[1]) / (math.factorial(x[0]) * math.factorial(x[1]))
    
    def cond(A):
        def cond_(conditional):
            c = np.sum([A**k/math.factorial(k) for k in range(m+1-conditional)])
            p_func = lambda x : (A**x/math.factorial(x))/c
            ps = [p_func(x) for x in range(m+1-conditional)]
            ps = np.pad(ps, (0,conditional))
            sample = np.random.choice(range(m+1), p = ps)
            return sample
        return cond_
    
    chain = gibbs_sampler(dist, cond(A1), cond(A2), x0, n_iter = chain_length)
    
    ps = get_multivariate_dist(A1, A2, m)
    
    obs = np.zeros((m+1,m+1))
    for element in chain:
        obs[element[0], element[1]] += 1
    obs /= np.sum(obs)
    
    # plot the two histograms
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(obs, origin = 'lower')
    axs[0].set_aspect('equal')
    axs[0].set_title("Observed")
    axs[1].imshow(ps, origin = 'lower')
    axs[1].set_aspect('equal')
    axs[1].set_title("Expected")
    
    fig.tight_layout()
    fig.suptitle("Gibbs Sampler")
    fig.savefig("gibbs2d.png")
    
    obs = extract_lower_triangle(obs)
    ps = extract_lower_triangle(ps)
    
    p = stats.chisquare(obs, ps)[1]
    
    plt.show()
    
    print(f"(Gibbs) Chi-square test p-value: {p}")
    
def ex3():
    
    # (a), generate pairs of theta and phi
    n = 10
    rho = 0.5
    xi_gamma = \
        np.random.multivariate_normal([0,0],
                                      [[1, rho],
                                       [rho, 1]], n)
    theta_phi = np.exp(xi_gamma)

    # (b) generate X_i
    X_i = np.random.normal(theta_phi[:,0], theta_phi[:,1])
    
    print(X_i)
    # (c) derive the posterior distribution
    
    """
    P(theta,phi|X) = P(X|theta,phi)P(theta,phi) / P(X)
    P(X) = sum_theta,phi P(X|theta,phi)P(theta,phi)
    P(theta,phi|X) = P(X|theta,phi)P(theta,phi) /
                     sum_theta,phi P(X|theta,phi)P(theta,phi)
    """
            
    
    # (d) generate MCMC samples
    # (e) repeat hte experiment

if __name__ == "__main__":
    
    ex1()
    ex2_1()
    ex2_2()
    ex2_3()
    ex3()
    
    #ex2
    
