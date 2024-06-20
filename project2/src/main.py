from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from models import SIR, CTMC, Q_SIR


OUTPUT_DIR = Path('./outputs/')


beta = 0.3
gamma = 0.1
population = 1000
initial_infected = 1



# theoretical model
model = SIR(beta=0.3, gamma=0.1, population=1000, initial_infected=1)
trajectory = model.simulate()

fig, ax = plt.subplots()
for i, label in enumerate(['Susceptible_theoretic', 'Infected_theoretic', 'Recovered_theoretic']):
    ax.plot(trajectory[:, i], label=label)


# stochastic model
get_Q = Q_SIR(
    beta=beta,
    gamma=gamma)

pi_0 = np.zeros(3, dtype=int)
pi_0[1] = initial_infected
pi_0[0] = population - pi_0.sum()

model = CTMC(get_Q, pi_0)
times, state_trajectory = model.simulate(
    continue_simulation = lambda state: state[1],
    tqdm_update = lambda delta_state: delta_state[2],
    tqdm_total = population - initial_infected
)


for i, (label, trajectory) in enumerate(zip(
    ('Susceptible_simulated', 'Infected_simulated', 'Recovered_simulated'),
    state_trajectory.T
)):
    ax.plot(times, trajectory, label=label)
ax.set_title(f"Time simulated={int(times[-1]):_}, recovered={state_trajectory[-1,2]:_}")


# plot
ax.legend()
ax.set_xlabel('Days')
ax.set_ylabel('Population')
plt.show()
