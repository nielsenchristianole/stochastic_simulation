
import numpy as np
import scipy.stats as stats

def pareto_sample(beta, k, n = 10_000):
    U = np.random.uniform(size=n)    
    return beta * U**(-1/k)
    
# def pareto(beta, k):
#     u = np.random.rand()
#     return beta / (u ** (1/k))

def pareto_mean(beta, k):
    return (k*beta)/(k-1)

def pareto_variance(beta, k):
    return (k * beta**2) / ((k-2)*((k-1)**2))

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
    n_bootstrap = 100
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
    n = 15
    
    # Bootstrap
    n_bootstrap = 100
    S2_hat = 0
    X_hat = np.mean(X_i)
    
    for i in range(n_bootstrap):
        subsample = np.random.choice(X_i, n)
        
        S2_hat += np.sum((subsample - X_hat)**2)/(n-1)
        
    S2_hat /= n_bootstrap
    
    return S2_hat

def ex3():
    
    beta = 1
    k = 1.05

    N_pareto = 200
    n_bootstrap = 1000
    r = 100
    
    mean_true = pareto_mean(beta, k)
    median_true = k*(2**(1/beta))
    pareto = pareto_sample(beta, k, N_pareto)
    
    median_estimates = np.zeros(n_bootstrap)
    mean_estimates = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        subsample = np.random.choice(pareto, r)
        
        median_estimates[i] = np.median(subsample)
        mean_estimates[i] = np.mean(subsample)
        
    median_variance = np.var(median_estimates)
    median = np.mean(median_estimates)
    mean_variance = np.var(mean_estimates)
    mean = np.mean(mean_estimates)
    
    
    print(f"Mean, estimate|true: {mean}|{mean_true}, variance: {mean_variance}")
    print(f"Median, estimate|true: {median}|{median_true}, variance: {median_variance}")
    # return median_estimates, median_variance
    
if __name__ == "__main__":
    
    
    # pareto = pareto_sample(1, 2.05)
    # print(np.mean(pareto))
    # print(np.var(pareto))
    # print(pareto_mean(1, 2.05))
    # print(pareto_variance(1, 2.05))
    
    print("Exercise 8")
    print("(1)")
    print(ex1())
    print("(2)")
    print(ex2())
    print("(3)")
    ex3()