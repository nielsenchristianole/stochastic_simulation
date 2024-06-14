
import numpy as np
import scipy.stats as stats

def ex1():
    """
    Exercise 13 from Ross chapter 8

        (a) The bootstrap approach can be used to estimate p
        by having a large number of samples from the distribution
        and then calculating the mean of the statistic from a collection
        of sub-samples.
        In the case of the problem, we would both have to estimate the mean
        and the probability that:
            a < X_i /n - mu < b
        E.i. the proportion of the subsamples that lies within the interval.
        In the case where we have limited data, we can sample with replacement
    """
    
    # (b). Estimate p using the bootstrap approach
    
    X_i = [56, 101, 78, 67, 93, 87, 64, 72, 80, 69]
    n = len(X_i)
    
    # Bootstrap
    n_bootstrap = 10000
    p_hat = 0
    
    a, b = -5, 5
    
    mu = np.mean(X_i)
    
    for i in range(n_bootstrap):
        subsample = np.random.choice(X_i, n)
        
        p_hat += a < np.sum(subsample)/n - mu < b

    p_hat /= n_bootstrap
    
    return p_hat

def ex2():
    """
    Estimate the variance Var(S**2) by simulation
    """
    
    X_i = [5,4,9,6,21,17,11,20,7,10,21,15,13,16,8]
    n = len(X_i)
    
    # Bootstrap
    n_bootstrap = 10000
    S2_hat = 0
    
    for i in range(n_bootstrap):
        subsample = np.random.choice(X_i, n)
        
        S2_hat += np.var(subsample, ddof=1)
        
    S2_hat /= n_bootstrap
    
    return S2_hat

def ex3():
    
    n = 200
    beta = 1.05
    k = 1

    median_true = k*(2**(1/beta))
    pareto = (np.random.pareto(beta, n) + 1)*k
    
    n_bootstrap = 200
    
    median_estimate = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        subsample = np.random.choice(pareto, n)
        
        median_estimate[i] = np.median(subsample)
        
    median_variance = np.var(median_estimate)
    median_estimate = np.mean(median_estimate)
    
    print(f"True median: {median_true}")
    return median_estimate, median_variance
    
if __name__ == "__main__":
    print(ex1())
    print(ex2())
    print(ex3())