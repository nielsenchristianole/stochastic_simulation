from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from models import CTMC, Q_custom


@dataclass
class ContinueSimulation:
    """
    class that will be used to continue the simulation until the number of infected is 0
    also keeps track of the maximum number of infections
    """
    max_infections: int = 0

    def __call__(self, state: np.ndarray) -> bool:
        current_infections = state[2]
        self.max_infections = max(self.max_infections, current_infections)
        return current_infections > 0


if __name__ == '__main__':

    do_simulation = True
    OUTPUT_DIR = Path('./outputs/')
    out_name = 'custom'

    # simulation parameters
    min_num_simulations_events = 4_000 # will condition the run to have at least this number of events before num infected is 0. Sampling via rejection sampling
    max_num_events = 100_000_000
    max_temporal_resolution = 1 # we want to keep track of the state every day

    # population parameters
    max_population: int | None = 7_000_000
    initial_population = 5_903_000 # 2022 DK population
    initial_infected = 100_000

    # set initial population
    pi_0 = np.zeros(14, dtype=int)
    pi_0[2] = initial_infected
    pi_0[1] = initial_population - pi_0.sum()

    # life/death parameters
    fertility_rate = 57_469 / initial_population / 365 # 2023 DK fertility rate
    healthy_death_rate = 58_384 / initial_population / 365 # 2023 DK death rate

    # non-vaccinated parameters
    recovery_rate = 1 / 14 # 14 days to recover
    infection_rate = 1.8 * recovery_rate # infected people infect 1.8 over recovery period
    infection_death_rate = 5 * healthy_death_rate # 5 times more likely to die if infected
    re_susceptibility_rate = 1 / (365 / 6) # takes 2 months to become susceptible again

    # vaccinated parameters
    time_to_vaccinate = 2 * 365 # 2 years to vaccinate everyone
    vaccination_rates = (0.05 / time_to_vaccinate, 0.30 / time_to_vaccinate, 0.65 / time_to_vaccinate) # 5%, 30%, 65% vaccination distribution
    vaccinated_infection_rates = (infection_rate / 100, infection_rate / 6, infection_rate / 2) # 0, 1/6, 1/2 of infection rate for vaccinated
    vaccinated_infection_death_rates = (1.02 * healthy_death_rate, 2 * healthy_death_rate, 4.5 * healthy_death_rate) # 1, 2, 4.5 times more likely to die if infected, 5 with vaccination
    vaccinated_recovery_rates = (10 * recovery_rate, 2 * recovery_rate, 1.3 * recovery_rate) # 10, 2, 1.3 times faster to recover
    vaccinated_re_susceptibility_rates = (re_susceptibility_rate / 10, re_susceptibility_rate / 2, re_susceptibility_rate / 1.3) # 10, 2, 1.3 times longer to become susceptible again


    state_descriptions = ['unborn',
        'susceptible', 'infected', 'recovered',
        'vaccinated_v1', 'infected_v1', 'recovered_v1',
        'vaccinated_v2', 'infected_v2', 'recovered_v2',
        'vaccinated_v3', 'infected_v3', 'recovered_v3',
        'deceased']


    get_Q = Q_custom(
        fertility_rate=fertility_rate,
        healthy_death_rate=healthy_death_rate,
        infection_rate=infection_rate,
        infection_death_rate=infection_death_rate,
        recovery_rate=recovery_rate,
        re_susceptibility_rate=re_susceptibility_rate,
        vaccination_rates=vaccination_rates,
        vaccinated_infection_rates=vaccinated_infection_rates,
        vaccinated_infection_death_rates=vaccinated_infection_death_rates,
        vaccinated_recovery_rates=vaccinated_recovery_rates,
        vaccinated_re_susceptibility_rates=vaccinated_re_susceptibility_rates,
        max_population=max_population)

    model = CTMC(get_Q, pi_0)

    if do_simulation:
        continue_simulation = ContinueSimulation()
        times, state_trajectory, transition_count = model.simulate(
            max_num_events=max_num_events,
            continue_simulation=continue_simulation,
            rejection_sample_num_min_events=min_num_simulations_events,
            return_transition_count=True,
            max_temporal_resolution=max_temporal_resolution)
    else:
        resuts = np.load(OUTPUT_DIR / f'{out_name}_simulation_results.npy', allow_pickle=True).item()['results']
        times = resuts['times']
        state_trajectory = resuts['state_trajectory']
        transition_count = resuts['transition_count']
        # times, state_trajectory, transition_count = model.continue_simulation(
        #     times[:-10],
        #     state_trajectory[:-10],
        #     max_num_events=max_num_events,
        #     continue_simulation=ContinueSimulation(),
        #     rejection_sample_num_min_events=min_num_simulations_events,
        #     return_transition_count=True,
        #     max_temporal_resolution=max_temporal_resolution)

    fig, ax = plt.subplots()
    for i, (label, trajectory) in enumerate(zip(
        state_descriptions,
        state_trajectory.T
    )):
        if label in ['unborn', 'deceased']:
            continue
        ax.plot(times, trajectory, label=label)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_title(f"Time simulated={int(times[-1]):_}" + (f'max_infections={continue_simulation.max_infections:_}' if do_simulation else ''))
    ax.set_xlabel('Days')
    ax.set_ylabel('Population')

    OUTPUT_DIR.mkdir(exist_ok=True)
    if do_simulation:
        simulation_results: dict[str, Any] = {
            'parameters': {
                'max_population': max_population,
                'initial_population': initial_population,
                'initial_infected': initial_infected,
                'fertility_rate': fertility_rate,
                'healthy_death_rate': healthy_death_rate,
                'recovery_rate': recovery_rate,
                'infection_rate': infection_rate,
                'infection_death_rate': infection_death_rate,
                're_susceptibility_rate': re_susceptibility_rate,
                'time_to_vaccinate': time_to_vaccinate,
                'vaccination_rates': vaccination_rates,
                'vaccinated_infection_rates': vaccinated_infection_rates,
                'vaccinated_infection_death_rates': vaccinated_infection_death_rates,
                'vaccinated_recovery_rates': vaccinated_recovery_rates,
                'vaccinated_re_susceptibility_rates': vaccinated_re_susceptibility_rates},
            'results': {
                'times': times,
                'state_trajectory': state_trajectory,
                'transition_count': transition_count,
                'max_infections': continue_simulation.max_infections}}
        np.save(OUTPUT_DIR / f'{out_name}_simulation_results.npy', simulation_results)
    plt.savefig(OUTPUT_DIR / f'{out_name}.pdf')
    plt.show()
