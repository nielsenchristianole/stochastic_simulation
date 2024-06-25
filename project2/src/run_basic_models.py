from pathlib import Path

import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models import SIR, CTMC, Q_SIR, Q_SIRVD, Q_SIS


OUTPUT_DIR = Path('./outputs/')


# will condition the run to have at least this number of events before num infected is 0. Sampling via rejection sampling
min_num_simulations_events = 100
population = 1000
initial_infected = 1
# for the SIR and SIS model
beta = 0.3 # infection rate
gamma = 0.1

season_weights = [[1,1], [0.5,1.5], [1,1], [1.5, 0.5]] # seasonality weights: spring, summer, fall, winter (beta, gamma)

# for the SIS model
num_sim_events = 90_000



def run_SIRVD(alpha, nu, mu, psi, resus_V, resus_R):
    get_Q = Q_SIRVD(
        alpha=alpha,
        nu=nu,
        mu=mu,
        psi=psi,
        resus_V=resus_V,
        resus_R=resus_R)

    pi_0 = np.zeros(5, dtype=int)
    pi_0[1] = initial_infected
    pi_0[0] = population - pi_0.sum()
    
    model = CTMC(get_Q, pi_0)

    times, state_trajectory = model.simulate(
        continue_simulation = lambda state: state[1],
        tqdm_update = lambda delta_state: delta_state[2:].sum(),
        tqdm_total = population - initial_infected,
        rejection_sample_num_min_events=min_num_simulations_events)
    return times, state_trajectory

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

    times, state_trajectory = model.simulate(
        continue_simulation = lambda state: state[1],
        tqdm_update = lambda delta_state: delta_state[2],
        tqdm_total = population - initial_infected,
        rejection_sample_num_min_events=min_num_simulations_events)

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
    plt.savefig(OUTPUT_DIR / 'sir_simulation1.pdf')
    plt.show()


# Q_SIRVD model
if True:
    # for the SIRVD model
    alpha, mu, psi = 0.01, 0.001, 0.001 # infection, recovery, and fatality
    resus_V = 1 / (365 / 6) # takes 2 months to become susceptible again
    resus_R = 1 / (365 * 2) # takes 2 years to become susceptible again
    
    # nus = [0.1,0.01,0.001,0.0001]
    # nus = [0.001]
    nus = [0.0001,0.001,0.01,0.1]
    num_bootstraps = 100
    
    for nu in nus:
        bootstrap_state_dead = []
        for i in tqdm.tqdm(range(num_bootstraps)):
            times, state_trajectory = run_SIRVD(alpha, nu, mu, psi, resus_V, resus_R)
            
            bootstrap_state_dead.append(state_trajectory[-1, -1])
            if i == 0:
                fig, ax = plt.subplots()
                for i, (label, trajectory) in enumerate(zip(
                    ('Susceptible', 'Infected', 'Recovered', 'Vacinated', 'Dead'),
                    state_trajectory.T
                )):
                    ax.plot(times, trajectory, label=label)
                    
                ax.plot(times, state_trajectory[:, :-1].sum(axis=1), label='Population')
                ax.set_title(f"SIRVD model, vaccination rate={nu}")
                
                ax.legend()
                ax.set_xlabel('Years')
                num_years = int(times[-1] / 365)
                num_ticks = 10
                ax.set_xticks(np.linspace(0, times[-1], num_ticks), labels=np.round(np.linspace(0, num_years, num_ticks),1))
                ax.set_ylabel('Population')
                plt.savefig(OUTPUT_DIR / f"SIRVD_vRate{str(nu).replace(".","-")}.pdf")
                plt.close()
                plt.clf()
        
        np.save(OUTPUT_DIR / f"SIRVD_vRate{str(nu).replace(".","-")}_bootstrap.npy", bootstrap_state_dead)


# Q_SIS model
if True:
    get_Q = Q_SIS(
        beta=beta,
        gamma=gamma,
        season_weights = season_weights)

    pi_0 = np.zeros(2, dtype=int)
    pi_0[1] = initial_infected
    pi_0[0] = population - pi_0.sum()
    
    model = CTMC(get_Q, pi_0)

    times, state_trajectory = model.simulate(
        max_num_events=num_sim_events,
        continue_simulation = lambda state: state[1],
        rejection_sample_num_min_events=min_num_simulations_events)

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