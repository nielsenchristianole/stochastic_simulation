from typing import Callable

import tqdm
import numpy as np


class CTMC():
    """
    Continoues Time Markov Chain SIR model
    """

    def __init__(
        self,
        get_Q: Callable[[np.ndarray], np.ndarray],
        pi_0: np.ndarray
    ) -> None:
        """
        Initialize the SIR model

        Parameters
        ----------
        get_Q: Callable[[np.ndarray], np.ndarray]
            Function that takes the current state and calculates the transition matrix
        pi_0: np.ndarray
            The initial state
            len(pi_0) == num_states
            should be population distribution and sum to population
        """
        assert pi_0.dtype == int, "Initial state should be integers of counts"

        # initialize
        self.pi_0 = pi_0
        self.population = pi_0.sum()
        self.num_states = len(pi_0)
        self.get_Q = get_Q

    def next_event(self, current_state) -> tuple[float, int, int]:
        """
        Sample the next event
        """
        Q = self.get_Q(current_state)

        rates = -np.diag(Q)
        transition_prob = Q
        non_arbsorbing_mask = rates > 0
        transition_prob[non_arbsorbing_mask] /= rates[non_arbsorbing_mask, None]
        np.fill_diagonal(transition_prob, ~non_arbsorbing_mask)

        # the rate of the next event is the rate of ant event happening
        event_rate = np.sum(rates)

        # we sample event and time
        time_to_next_event =  np.random.exponential(1 / event_rate)
        before_state = np.random.choice(self.num_states, p=rates / event_rate)
        next_state = np.random.choice(self.num_states, p=transition_prob[before_state])
        return time_to_next_event, before_state, next_state

    def simulate(
        self,
        max_num_events: int | None = None,
        *,
        continue_simulation: Callable[[np.ndarray], bool] | None = None,
        tqdm_update: Callable[[np.ndarray], int] | None = None,
        tqdm_total: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the SIR model

        parameters
        ----------
        max_num_events: how many events to simulate
        continue_simulation: function that takes the current state and returns true if the simulation should continue
        tqdm_update: function that takes the difference in states and returns how much the simulation progressed
        tqdm_total: how much simulation to do
        """
        assert (max_num_events is not None) or (continue_simulation is not None), "We need some way to tell when to end the simulation"

        # init for simulation
        times = np.zeros((1,))
        state_trajectory = self.pi_0.copy()[None, :]

        current_time = 0
        current_state = self.pi_0.copy()

        if max_num_events is not None:
            pbar = tqdm.tqdm(desc="Simulating...", total=max_num_events, leave=False)
            tqdm_update = lambda _: 1
        else:
            pbar = tqdm.tqdm(desc="Simulating...", total=tqdm_total, leave=False)

        i = 1
        max_length = 1
        while (((continue_simulation is None) or continue_simulation(current_state)) and
               ((max_num_events is None) or (i < max_num_events))):

            # make room for more simulation
            if max_length <= i:
                max_length *= 2
                times = np.concatenate((times, np.zeros_like(times)), axis=0)
                state_trajectory = np.concatenate((state_trajectory, np.zeros_like(state_trajectory)), axis=0)

            # take a step in the simulation
            time_to_next_event, before_state, next_state = self.next_event(current_state)
            current_time += time_to_next_event
            delta_state = np.zeros_like(self.pi_0)
            delta_state[before_state] = -1
            delta_state[next_state] = 1
            current_state += delta_state
            times[i] = current_time
            state_trajectory[i] = current_state.copy()
            pbar.update(tqdm_update(delta_state))
            i += 1

        pbar.close()

        # no reason to keep empty arrays
        times = times[:i]
        state_trajectory = state_trajectory[:i]

        return times, state_trajectory


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.models.ctmc_model_factories import Q_SIR

    beta = 0.3
    gamma = 0.1
    population = 1000
    initial_infected = 1


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

    fig, ax = plt.subplots()
    for i, (label, trajectory) in enumerate(zip(
        ('Susceptible', 'Infected', 'Recovered'),
        state_trajectory.T
    )):
        ax.plot(times, trajectory, label=label)
    ax.set_title(f"Time simulated={int(times[-1]):_}, recovered={state_trajectory[-1,2]:_}")
    
    ax.legend()
    ax.set_xlabel('Days')
    ax.set_ylabel('Population')
    plt.show()
