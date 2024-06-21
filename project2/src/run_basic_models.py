from pathlib import Path

import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models import SIR, CTMC, Q_SIR, Q_SIRVD, Q_custom, Q_SIS


OUTPUT_DIR = Path('./outputs/')


# will condition the run to have at least this number of events before num infected is 0. Sampling via rejection sampling
min_num_simulations_events = 400

# for the SIR and SIS model
beta = 0.3
gamma = 0.1
population = 1000
initial_infected = 1

# for the SIS model
num_sim_events = 10_000

# for the SIRVD model
alpha, nu, mu, psi = 0.3, 0.0005, 0.03, 0.07


# SIR model
if True:
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

    pbar = tqdm.tqdm(desc=f'Simulating SIR model, rejection sample, length > {min_num_simulations_events}', leave=False)
    while True:
        times, state_trajectory = model.simulate(
            continue_simulation = lambda state: state[1],
            tqdm_update = lambda delta_state: delta_state[2],
            tqdm_total = population - initial_infected)
        pbar.update(1)
        pbar.set_postfix({'last simulation length': len(times)})
        if len(times) > min_num_simulations_events:
            break
    pbar.close()


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
    plt.savefig(OUTPUT_DIR / 'SIR.pdf')
    plt.show()


# Q_SIRVD model
if True:

    get_Q = Q_SIRVD(
        alpha=alpha,
        nu=nu,
        mu=mu,
        psi=psi,)

    pi_0 = np.zeros(5, dtype=int)
    pi_0[1] = initial_infected
    pi_0[0] = population - pi_0.sum()
    
    model = CTMC(get_Q, pi_0)

    pbar = tqdm.tqdm(desc=f'Simulating SIRVD model, rejection sample, length > {min_num_simulations_events}', leave=False)
    while True:
        times, state_trajectory = model.simulate(
            continue_simulation = lambda state: state[1],
            tqdm_update = lambda delta_state: delta_state[2:].sum(),
            tqdm_total = population - initial_infected)
        pbar.update(1)
        pbar.set_postfix({'last simulation length': len(times)})
        if len(times) > min_num_simulations_events:
            break
    pbar.close()

    fig, ax = plt.subplots()
    for i, (label, trajectory) in enumerate(zip(
        ('Susceptible', 'Infected', 'Recovered', 'Vacinated', 'Dead'),
        state_trajectory.T
    )):
        ax.plot(times, trajectory, label=label)
    ax.set_title(f"Time simulated={int(times[-1]):_}")
    
    ax.legend()
    ax.set_xlabel('Days')
    ax.set_ylabel('Population')
    plt.savefig(OUTPUT_DIR / 'SIRVD.pdf')
    plt.show()


# Q_SIS model
if True:
    get_Q = Q_SIS(
        beta=beta,
        gamma=gamma)

    pi_0 = np.zeros(2, dtype=int)
    pi_0[1] = initial_infected
    pi_0[0] = population - pi_0.sum()
    
    model = CTMC(get_Q, pi_0)

    pbar = tqdm.tqdm(desc=f'Simulating SIS model, rejection sample, length > {min_num_simulations_events}', leave=False)
    while True:
        times, state_trajectory = model.simulate(
            max_num_events=num_sim_events,
            continue_simulation = lambda state: state[1])
        pbar.update(1)
        pbar.set_postfix({'last simulation length': len(times)})
        if len(times) > min_num_simulations_events:
            break
    pbar.close()

    fig, ax = plt.subplots()
    for i, (label, trajectory) in enumerate(zip(
        ('Susceptible', 'Infected'),
        state_trajectory.T
    )):
        ax.plot(times, trajectory, label=label)
    ax.set_title(f"Time simulated={int(times[-1]):_}")
    
    ax.legend()
    ax.set_xlabel('Days')
    ax.set_ylabel('Population')
    plt.savefig(OUTPUT_DIR / 'SIS.pdf')
    plt.show()