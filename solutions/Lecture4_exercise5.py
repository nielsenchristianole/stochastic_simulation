
import numpy as np

def crude_monte_carlo(func, n):
    """
    Estimate the integral by using the crude Monte Carlo method.
    The function is crude since we just sample uniformly
    and then take the mean of the function evaluated at the samples.
    """

    U = np.random.uniform(0, 1, n)
    
    X = func(U)
    
    return np.mean(X)

def antithetic_estimate(func, n):
    """
    Estimating the integral by unsing antuthetic variates.
    The function works by sampling from both sides of the uniform distibution
    U1 is uniform, so U2 = 1 - U1 is also uniform.
    The estimate:
        [func(U1) + func(U2)] / 2
    is then used since both func(U1) and func(U2) is unbiased estimates of the integral.
    
    We have to evaluate func twice. We therefore do n/2 to 
    make a fair comparison.
    """
    
    U1 = np.random.uniform(0, 1, n//2)
    U2 = 1 - U1
    
    X = (func(U1) + func(U2)) / 2
    
    return np.mean(X)

def control_variates(func, n, c, optimize_c = False):
    """
    Estimate the integral using control variates.
        Z = X + c(Y - E[Y])
    This estimator is unbiased since:
        E[Z] = E[X] + c(E[Y] - E[Y]) = E[X]
    We can choose c to minimize the variance of Z. This can be derived by:
        Var[Z] = Var[X] + c^2 Var[Y] - 2c Cov[X, Y]
    The optimal c is:
        c = - Cov[X, Y] / Var[Y]
    
    The covariance is unknown but can be found by using an initial guess of c,
    then calculate the covariance and update c.
    It can be done untill convergence or just by using a subset of the number of samples.
        If optimize_c = True, n//10 samples are used to optimize c.
        
    The uniform distribution is used here as Y.
    We know that:
        E[Y] = 1/2
        Var[Y] = 1/12
    """
        
    U = np.random.uniform(0, 1, n)
    
    X = func(U)
    Z = X + c * (U - 1/2)

    if optimize_c:
        Y = U
        c = - np.cov(X, Y)[0, 1] / (1/12)
        return control_variates(func, n, c, optimize_c = False)
    
    return np.mean(Z)

def stratified_sampling(func, n, strata = [0, 1]):
    """
    Estimate the integral by using stratified sampling.
    The idea is to divide the interval into n strata and sample uniformly from each strata.
    Using the default strate = [0, 1] is equivalent to the crude Monte Carlo method.
    """
    
    n_strata = len(strata) - 1
    n_per_strata = n // n_strata
    
    X = np.zeros(n_strata)
    for i in range(n_strata):
        U = np.random.uniform(strata[i], strata[i+1], n_per_strata)
        X[i] = np.mean(func(U))
        
    return np.mean(X)

def importance_sampling():
    
    lambda_ = 1
    f = lambda x : 1
    p = lambda x : np.exp(x)
    q = lambda x : lambda_ * np.exp(-lambda_ * x)

    n = 1000
    U = np.random.exponential(1, n)
    X = f(U) * p(U) / q(U)
    
    return np.mean(X) 
    

if __name__ == "__main__":
    
    n = 1000
    func = np.exp
    facit = np.e - 1
    
    res = crude_monte_carlo(func, n)
    print("CRUDE MONTE CARLO | FACIT | DIFFERENCE")
    print(res, facit, res - facit)
    
    res = antithetic_estimate(func, n)
    print("ANTITHETIC ESTIMATE | FACIT | DIFFERENCE")
    print(res, facit, res - facit)
    
    res = control_variates(func, n, -1, optimize_c = True)
    print("CONTROL VARIATES (OPTIMIZED) | FACIT | DIFFERENCE")
    print(res, facit, res - facit)
    
    res = control_variates(func, n, -1, optimize_c = False)
    print("CONTROL VARIATES (NOT OPTIMIZED) | FACIT | DIFFERENCE")
    print(res, facit, res - facit)
    
    res = stratified_sampling(func, n)
    print("STRATIFIED SAMPLING (2 STRATA) | FACIT | DIFFERENCE")
    print(res, facit, res - facit)
    
    
    
    
    
    