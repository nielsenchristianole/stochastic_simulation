
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def simulated_annealing(init_state, U, neighbour_func, T_schedular, k_max = 1000):
    
    s = init_state
    
    P = lambda x, y, T : np.exp(-(U(x) - U(y))/T)
    
    for k in range(k_max):
        
        T = T_schedular(k)
        s_new = neighbour_func(s)
        
        if P(s, s_new, T) > np.random.rand():
            s = s_new
    return s

def generate_random_problem(n = 10):
    return np.random.rand(n, 2)
    
def generate_trivial_problem(n = 10):
    
    points = []
    
    step = np.pi*2/n
    
    for i in range(n):
        points.append([np.cos(i*step), np.sin(i*step)])
        
    points = (np.array(points) + 1) / 2
    
    return points

def init_state(type = "random", n = 10):
    
    if type == "trivial":
        points = generate_trivial_problem(n)
    else:
        points = generate_random_problem(n)
        
    # Create a n x n matrix of connections
    state = np.zeros((n, n))
    for i in range(n - 1):
        state[i, i+1] = 1
    state[-1, 0] = 1
    
    return points, state

def neighbour_func(state):
    """
    Swap the connections between two random points
    """
    idx0 = np.random.randint(0, len(state)-2)
    idx1 = np.random.randint(idx0+2, len(state))
    
    in_idx0 = np.argwhere(state[idx0] == 1)
    out_idx0 = np.argwhere(state[:, idx0] == 1)
    in_idx1 = np.argwhere(state[idx1] == 1)
    out_idx1 = np.argwhere(state[:, idx1] == 1)
    
    new_state = np.copy(state)
    
    new_state[idx0, in_idx0] = 0
    new_state[out_idx0, idx0] = 0
    new_state[idx1, in_idx1] = 0
    new_state[out_idx1, idx1] = 0
    
    new_state[idx1, in_idx0] = 1
    new_state[out_idx0, idx1] = 1
    new_state[idx0, in_idx1] = 1
    new_state[out_idx1, idx0] = 1
    
    return new_state
    
def visualize_state(points, state, color = "blue"):
    
    for i in range(len(state)):
        for j in range(len(state)):
            if state[i, j] == 1:
                plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color)
        
    plt.scatter(points[:,0], points[:,1])
    plt.axis('equal')

if __name__ == "__main__":
    
    T1 = lambda k :  1/np.sqrt(1+k)
    T2 = lambda k : -np.log(k+1)
    
    points, state = init_state("trivial")
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    state = neighbour_func(state)
    visualize_state(points, state, "blue")
    
    plt.show()
    
    # points, state = init_state("trivial")
    
    # plt.scatter(points_trivial[:,0], points_trivial[:,1])
    # plt.scatter(points_random[:,0], points_random[:,1])
    # plt.axis('equal')
    # plt.show()