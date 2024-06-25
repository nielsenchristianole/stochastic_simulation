

import numpy as np
import scipy.stats as stats
import math

def pareto_sample(k, beta = 1, size = 10_000):
    U = np.random.uniform(size = size)    
    return beta * np.power(U,-1/k)

def pareto_mean(k, beta = 1):
    return (k*beta)/(k-1)


def simulate(inter_arrival_dist, service_dist,
             inter_arrival_params, service_params,
             batch_size, num_batches, num_services):
    num_customers = batch_size * num_batches

    arrivals = np.cumsum(inter_arrival_dist(**inter_arrival_params, size = num_customers))
    service_times = service_dist(**service_params, size = num_customers)

    num_blocked = np.zeros(num_batches)
    free_at = np.zeros(num_services)
    total_inter_arrival_times = np.zeros(num_batches)

    for i, time in enumerate(arrivals):
        
        num_free = len(free_at[free_at <= time])
        if i != 0:
            total_inter_arrival_times[i // batch_size] += arrivals[i] - arrivals[i - 1]

        if num_free == 0:
            num_blocked[i // batch_size] += 1
            continue
        
        free_at[np.argwhere(free_at <= time)[0][0]] = time + service_times[i]

    
    return num_blocked, total_inter_arrival_times

def hyperexponential(lam1, lam2, p1, size):
    U = np.random.random(1)
    
    out = np.zeros(size)
    for i in range(size):
        if U <= p1:
            out[i] = np.random.exponential(lam1)
        else:
            out[i] = np.random.exponential(lam2)
    return out

def erlang_B_formula(lambda_, scale, m):
    A = lambda_ * scale
    B = A**m / math.factorial(m)
    B = B / np.sum([A**i/math.factorial(i) for i in range(m+1)])
    return B

def control_variate_adjustment(X, Y, E_Y):
    cov_XY = np.cov(X, Y, ddof=1)[0, 1]
    var_Y = np.var(Y, ddof=1)
    c = cov_XY / var_Y
    X_star = X - c * (Y - E_Y)
    return X_star
        
if __name__ == "__main__":
    
    print("k=1.05, beta=0.38, mean service = 8")
    samples = pareto_sample(1.05, 0.38, 100_000)
    print(np.sort(samples)[-10:])
    print("k=2.05, beta=4.10, mean service = 8")
    samples = pareto_sample(2.05, 4.10, 100_000)
    print(np.sort(samples)[-10:])
        
# if __name__ == "__main__":
#     batch_size = 10_000
#     num_batches = 10
#     num_services = 10

#     # --- poisson arrival and exponential service ---

#     np.random.seed(1)
#     inter_arrival_dist = np.random.exponential
#     service_dist = np.random.exponential

#     inter_arrival_params = {"scale" : 1}
#     service_params = {"scale" : 8}

#     num_blocked, total_inter_arrival_times = simulate(inter_arrival_dist, service_dist,
#                            inter_arrival_params, service_params,
#                            batch_size, num_batches, num_services)
    
#     # Expected total inter-arrival time in each batch
#     E_total_inter_arrival_times = batch_size * inter_arrival_params["scale"]

#     X_pois = num_blocked / batch_size
#     Y = total_inter_arrival_times / batch_size

#     X_star = control_variate_adjustment(X_pois, Y, E_total_inter_arrival_times / batch_size)

#     print("Poisson arrival and Exponential service (original):", np.mean(X_pois), np.var(X_pois, ddof=1))
#     print("Poisson arrival and Exponential service (adjusted):", np.mean(X_star), np.var(X_star, ddof=1))
    
#     # Calculate the confidence interval
#     alpha = 0.05
#     z = stats.norm.ppf(1 - alpha/2)
#     CI = [np.mean(X_star) - z * np.sqrt(np.var(X_star, ddof=1) / num_batches),
#             np.mean(X_star) + z * np.sqrt(np.var(X_star, ddof=1) / num_batches)]
#     print("Confidence interval:", CI)
    
    
#     print("Exact Erlang B-formula: ", erlang_B_formula(inter_arrival_params["scale"],
#                                                        service_params["scale"],
#                                                        m = 10), "\n")
#     # --- erlang interarrival and exponential service ---
    
#     inter_arrival_dist = stats.erlang.rvs
#     service_dist = np.random.exponential
    
#     inter_arrival_params = {"a" : 1}
#     service_params = {"scale" : 8}

#     num_blocked, total_inter_arrival_times = simulate(inter_arrival_dist, service_dist,
#                             inter_arrival_params, service_params,
#                             batch_size, num_batches, num_services)

#     X_exp = num_blocked/batch_size
#     print("Erlang interarrival and exponential service", np.mean(X_exp))
    
#     # --- hyperexponential interarrival and exponential service ---
    
#     np.random.seed(1)
#     inter_arrival_dist = hyperexponential
#     service_dist = np.random.exponential
    
#     inter_arrival_params = {"lam1" : 0.8333, "lam2" : 5.0, "p1" : 0.8}
#     service_params = {"scale" : 8}
    
#     num_blocked, _ = simulate(inter_arrival_dist, service_dist,
#                             inter_arrival_params, service_params,
#                             batch_size, num_batches, num_services)
    
#     X_hyp = num_blocked/batch_size
#     print("Hyperexponential interarrival and exponential service", np.mean(X_hyp))
    
#     alpha = 0.05
#     z = stats.norm.ppf(1 - alpha/2)
#     diff = X_hyp - X_pois
#     CI = [np.mean(diff) - z * np.sqrt(np.var(diff, ddof=1) / num_batches),
#             np.mean(diff) + z * np.sqrt(np.var(diff, ddof=1) / num_batches)]
#     print("Confidence interval diff:", CI, "\n")
    
    
#     # --- poisson arrival and constant service ---
    
#     inter_arrival_dist = np.random.exponential
#     service_dist = lambda size : np.full(size, 8)
    
#     inter_arrival_params = {"scale" : 1}
#     service_params = {}
    
#     num_blocked, _ = simulate(inter_arrival_dist, service_dist,
#                             inter_arrival_params, service_params,
#                             batch_size, num_batches, num_services)
    
#     X = num_blocked/batch_size
#     print("Poisson arrival and constant service", np.mean(X))
#     print("Exact Erlang B-formula: ", erlang_B_formula(inter_arrival_params["scale"],
#                                                        8,
#                                                        m = 10), "\n")
#     print("CI:", np.percentile(X, 2.5), np.percentile(X, 97.5), "\n")
    
#     # --- poisson arrival and pareto service (1) ---
    
#     inter_arrival_dist = np.random.exponential
#     service_dist = pareto_sample
    
#     k = 1.05
#     beta = 8*(k-1)/k
#     inter_arrival_params = {"scale" : 1}
#     service_params = {"k" : k, "beta" : beta}
    
#     num_blocked, _ = simulate(inter_arrival_dist, service_dist,
#                             inter_arrival_params, service_params,
#                             batch_size, num_batches, num_services)
    
#     X = num_blocked/batch_size
#     print("Beta:", beta, "k:", k)
#     print("Poisson arrival and pareto service (1)", np.mean(X))
#     print("Exact Erlang B-formula: ", erlang_B_formula(inter_arrival_params["scale"],
#                                                        pareto_mean(service_params["k"],
#                                                                    service_params["beta"]),
#                                                        m = 10))
#     print("CI:", np.percentile(X, 2.5), np.percentile(X, 97.5), "\n")
    
#     # --- poisson arrival and pareto service (2) ---
    
#     inter_arrival_dist = np.random.exponential
#     service_dist = pareto_sample
    
#     k = 2.05
#     beta = 8*(k-1)/k
#     inter_arrival_params = {"scale" : 1}
#     service_params = {"k" : k, "beta" : beta}
    
#     num_blocked, _= simulate(inter_arrival_dist, service_dist,
#                             inter_arrival_params, service_params,
#                             batch_size, num_batches, num_services)
    
#     X = num_blocked/batch_size
#     print("Beta:", beta, "k:", k)
#     print("Poisson arrival and pareto service (2)", np.mean(X))
#     print("Exact Erlang B-formula: ", erlang_B_formula(inter_arrival_params["scale"],
#                                                        pareto_mean(service_params["k"],
#                                                                    service_params["beta"]),
#                                                        m = 10))
#     print("CI:", np.percentile(X, 2.5), np.percentile(X, 97.5), "\n")

#     # --- poisson arrival and weibull service ---
    
#     inter_arrival_dist = np.random.exponential
#     service_dist = np.random.weibull
    
#     inter_arrival_params = {"scale" : 1}
#     service_params = {"a" : 2}
    
#     num_blocked, _= simulate(inter_arrival_dist, service_dist,
#                             inter_arrival_params, service_params,
#                             batch_size, num_batches, num_services)
    
#     X = num_blocked/batch_size
#     print("Poisson arrival and weibull", np.mean(X))
#     print("Exact Erlang B-formula: ", erlang_B_formula(inter_arrival_params["scale"],
#                                                        stats.gamma.mean(service_params["a"]),
#                                                        m = 10), "\n")
    
#     # confidence interval
#     alpha = 0.05
    
#     z = stats.norm.ppf(1 - alpha/2)
#     CI = [np.mean(X) - z * np.sqrt(np.var(X, ddof=1) / num_batches),
#             np.mean(X) + z * np.sqrt(np.var(X, ddof=1) / num_batches)]  
#     print("Confidence interval:", CI)
    
    
    













# import numpy as np
# import tqdm
# import scipy.stats as stats
    

# class ServiceUnit:
    
#     def __init__(self, service_dist, dist_params):
        
#         self.occupied = False
#         self.time_till_free = 0
#         self.service_dist = service_dist
#         self.dist_params = dist_params
    
#     def check_in(self):
#         self.occupied = True
#         self.time_till_free = self.service_dist(**self.dist_params) + 1
        
#     def process(self):
#         if self.occupied:
#             self.time_till_free -= 1
#             if self.time_till_free <= 0:
#                 self.occupied = False
        
# class Simulator:
    
#     def __init__(self, arrival_dist, inter_arrival_dist,
#                  arrival_dist_params,
#                  inter_arrival_dist_params,
#                  out_file_name = "simulation.txt",
#                  num_batches = 10, batch_size = 10_000):
#         self.m = 10
#         self.scale = 10
#         self.lambda_ = 5
        
#         self.arrival_dist = arrival_dist
#         self.inter_arrival_dist = inter_arrival_dist
#         self.arrival_dist_params = arrival_dist_params
        
#         self.service_units = self._get_service_units(inter_arrival_dist, inter_arrival_dist_params)
        
#         self.time = 0
#         self.customers = 0
#         self.file_name = out_file_name
#         self.batch_size = batch_size
#         self.num_batches = num_batches
        
#         with open(self.file_name, 'w') as f:
#             f.write("time event\n")       
        
#     def simulate(self):
#         next_arrival_in = self.arrival_dist(**self.arrival_dist_params)
#         pbar = tqdm.tqdm(total = self.batch_size * self.num_batches)
#         while self.customers < self.batch_size * self.num_batches:
#             if next_arrival_in <= 0:
#                 next_arrival_in = self.arrival_dist(**self.arrival_dist_params)
#                 self._attempt_check_in()
#                 self.customers += 1
#                 pbar.update(1)
                
#             for unit in self.service_units:
#                 unit.process()
                
#             self.time += 1
#             next_arrival_in -= 1
            
#     def _get_service_units(self, dist, dist_params):
#         return [ServiceUnit(dist, dist_params) for _ in range(self.m)]

#     def _stream_to_file(self, data : str):
#         with open(self.file_name, 'a') as f:
#             f.write(f'{self.time} {data}\n')

#     def _attempt_check_in(self):
#         occupied_units = 0
#         for unit in self.service_units:
#             if unit.occupied:
#                 occupied_units += 1
#             else:
#                 unit.check_in()
#                 self._stream_to_file("check_in")
#                 break
#         if occupied_units == self.m:
#             self._stream_to_file("blocked")

# def hyperexponential(lam1, lam2, p1):
#     U = np.random.random(1)
    
#     if U <= p1:
#         return np.random.exponential(lam1)
#     else:
#         return np.random.exponential(lam2)

# if __name__ == "__main__":
    
#     # --- poisson arrival and exponential service ---
    
#     arrival_dist = np.random.poisson
#     arrival_dist_params = {"lam" : 1}
    
#     inter_arrival_dist = np.random.exponential
#     inter_arrival_dist_params = {"scale" : 10}
    
#     sim = Simulator(arrival_dist = arrival_dist,
#                     inter_arrival_dist = inter_arrival_dist,
#                     arrival_dist_params = arrival_dist_params,
#                     inter_arrival_dist_params = inter_arrival_dist_params,
#                     out_file_name = "poission_exponential.txt")
#     sim.simulate()
    
#     # --- poisson arrival and erlang service ---
    
#     arrival_dist = np.random.poisson
#     arrival_dist_params = {"lam" : 1}
    
#     inter_arrival_dist = stats.erlang.rvs
#     inter_arrival_dist_params = {"a" : 1}
    
#     sim = Simulator(arrival_dist = arrival_dist,
#                     inter_arrival_dist = inter_arrival_dist,
#                     arrival_dist_params = arrival_dist_params,
#                     inter_arrival_dist_params = inter_arrival_dist_params,
#                     out_file_name = "poission_erlang.txt")
#     sim.simulate()
    
#     # --- poisson arrival and hyperexponential service ---
    
#     arrival_dist = np.random.poisson
#     arrival_dist_params = {"lam" : 1}
    
#     inter_arrival_dist = hyperexponential
#     inter_arrival_dist_params = {"lam1" : 0.8333, "lam2" : 5.0, "p1" : 0.8}
    
#     sim = Simulator(arrival_dist = arrival_dist,
#                     inter_arrival_dist = inter_arrival_dist,
#                     arrival_dist_params = arrival_dist_params,
#                     inter_arrival_dist_params = inter_arrival_dist_params,
#                     out_file_name = "poission_hyperexpo.txt")
#     sim.simulate()
    
    