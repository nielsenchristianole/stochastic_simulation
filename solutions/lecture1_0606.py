
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import tqdm

def linear_congruential(x0, a, c, M):
    x = x0
    randint = []
    for _ in range(M):
        x = (a * x + c) % M
        randint.append(x/M)
    return np.array(randint)

def relative_prime(a, b):
    while math.gcd(a,b) != 1:
        b += 1
    return a, b

def xi_square_uniform(data, bins):
    hist, _ = np.histogram(data, bins=bins)
    expected = len(data)/bins
    
    assert expected > 5, f"Expected frequency should be greater than 5, currently {expected}"
    
    z = sum((hist - expected)**2 / expected)
    p = stats.chi2.cdf(z, bins-1)
    return p, 1-stats.chisquare(hist)[1]

def kolmogorov_smirnov(data):
    # sorted = np.sort(data)
    # n = len(sorted)
    # d_plus = np.max([(i+1)/n - sorted[i] for i in range(n)])
    # d_minus = np.max([sorted[i] - i/n for i in range(n)])
    
    # z = np.max([d_plus, d_minus])
    
    return 1-stats.kstest(data, 'uniform')[1]

def above_below_run_test(data):
    """
    Wald-Wolfowitz run test
    """
    median = np.median(data)
    above = 0
    below = 0
    run = 1
    
    for i in range(len(data)):
        if data[i] > median:
            above += 1
            if i > 0 and data[i-1] <= median:
                run += 1
        elif data[i] < median:
            below += 1
            if i > 0 and data[i-1] > median:
                run += 1
    
    mean = 2*above*below/(above+below) + 1
    variance = 2*above*below*(2*above*below - above - below) /\
               ((above+below)**2 * (above+below-1))
               
    # Calcualte p-value
    z = (run - mean) / math.sqrt(variance)
    p_value = 2*(1 - stats.norm.cdf(abs(z)))
    
    return p_value
    
def up_down_run_test(data):
    """
    The Up/Down test from Knuth's Art of Computer Programming
    """
    # assert len(data) > 4000, "Data should be at least 4000"
    
    A = np.array([[4529.4, 9044.9, 13568, 18091,  22615,  227892],
                  [9044.9, 18097,  27139, 36187,  45234,  55789],
                  [13568,  27139,  40721, 54281,  67852,  83685],
                  [18091,  36187,  54281, 72414,  90470,  111580],
                  [22615,  45234,  67852, 90470,  113262, 139476],
                  [27892,  55789,  83685, 111580, 139476, 172860]])
    
    B = np.array([1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])
    
    # Count consecutive increasing and decreasing values
    R = np.zeros(6)
    
    run_length = 0
    for i in range(1, len(data)):
        run_length += 1
        if data[i] <= data[i-1]:
            length = max(1, min(6, run_length))
            R[length-1] += 1
            run_length = 0
    R[min(6, run_length)] += 1
    
    n = len(data)
    z = (R-n*B).T @ A @ (R-n*B)/(n-6)
    
    p = 1-stats.chi2.cdf(z, 6)
    return p

def the_up_and_down_test(data):
    
    less_greater_seq = []
    for i in range(1, len(data)):
        if data[i] > data[i-1]:
            less_greater_seq.append(1)
        elif data[i] < data[i-1]:
            less_greater_seq.append(-1)

    num_runs = 1
    for i in range(1, len(less_greater_seq)):
        if less_greater_seq[i] != less_greater_seq[i-1]:
            num_runs += 1

    z = (num_runs-(2*len(data)-1)/3) /\
        math.sqrt((16*len(data)-29)/90)
    
    p = 2*(1 - stats.norm.cdf(abs(z)))
    return p

def correlation_test(data, lag=1):
    """
    Correlation cofficients test
    """
    corr = np.sum(data[:-lag]*data[lag:]) / len(data - lag)
    corr_coef = abs((corr - 0.25)/np.sqrt(7/(144*len(data))))
    p = 2*(1 - stats.norm.cdf(corr_coef))
    return p, stats.pearsonr(data[:-1], data[1:])[1]

if __name__ == "__main__":
    
    np.random.seed(2)
    runs = 500
    results = np.empty((runs, 6))
    
    for i in tqdm.tqdm(range(runs)):
        a = np.random.randint(100_000,1_000_000)
        c = np.random.randint(100_000,1_000_000)
        bins = 10
        samples = 10_000
        
        a, c = relative_prime(a, c)
        
        num = linear_congruential(1, a, c, samples)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].hist(num.flatten(), bins=bins, density=True)
        axs[1].plot(num, '.',markersize=1)
        plt.show()
        
        xi = xi_square_uniform(num, bins)
        kolo = kolmogorov_smirnov(num)
        run_I = above_below_run_test(num)
        run_II = up_down_run_test(num)
        run_III = the_up_and_down_test(num)
        corr = correlation_test(num)
        
        results[i] = np.array([xi[0],
                               kolo,
                               run_I,
                               run_II,
                               run_III,
                               corr[0]])
    
    print(f"Xi-square: {np.mean(results[:,0])}")
    print(f"Kolmogorov-Smirnov: {np.mean(results[:,1])}")
    print(f"Above-Below Run Test: {np.mean(results[:,2])}")
    print(f"Up-Down Run Test: {np.mean(results[:,3])}")
    print(f"The Up and Down Test: {np.mean(results[:,4])}")
    print(f"Correlation Test: {np.mean(results[:,5])}")
    
    fig, axs = plt.subplots(1,6, figsize=(20, 5))
    for i in range(6):
        axs[i].ecdf(results[:,i])
    
    plt.show()
        