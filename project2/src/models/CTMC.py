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
        self._rejection_sampling_pbar = None

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
        tqdm_total: int | None = None,
        return_transition_count: bool = False,
        max_temporal_resolution: float = 0.,
        rejection_sample_num_min_events: int | None = None,
        rejection_sample_num_max_tries: int = 100,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the SIR model

        If one wishes to gather information about the simulation, simple make `continue_simulation` a callable object
        that gathers the information and returns False when the simulation should stop.

        parameters
        ----------
        max_num_events: how many events to simulate
        continue_simulation: function that takes the current state and returns true if the simulation should continue
        tqdm_update: function that takes the difference in states and returns how much the simulation progressed
        tqdm_total: how much simulation to do
        return_transition_count: return the transition count matrix
        max_temporal_resolution: if multiple events happen within this time, only record the last event
        rejection_sample_num_min_events: minimum number of events to simulate, will keep simulating until this number is reached
        rejection_sample_num_max_tries: maximum number of tries to simulate, will raise an error if this number is reached
        """
        assert (max_num_events is not None) or (continue_simulation is not None), "We need some way to tell when to end the simulation"

        if (rejection_sample_num_max_tries is not None) and (rejection_sample_num_max_tries <= 0):
            self._rejection_sampling_pbar.close()
            self._rejection_sampling_pbar = None
            raise RuntimeError("Max number of tries reached")

        # init for simulation
        times = np.zeros((1,))
        state_trajectory = self.pi_0.copy()[None, :]

        current_time = 0
        current_state = self.pi_0.copy()
        if return_transition_count:
            num_states = len(self.pi_0)
            transition_count = np.zeros((num_states, num_states), dtype=int)

        if max_num_events is not None:
            pbar = tqdm.tqdm(desc="Simulating...", total=max_num_events, leave=False)
            tqdm_update = lambda _: 1
        else:
            pbar = tqdm.tqdm(desc="Simulating...", total=tqdm_total, leave=False)

        i = 0
        num_events_counter = 0
        time_since_last_event = 0
        max_length = 1
        while (((continue_simulation is None) or continue_simulation(current_state)) and
               ((max_num_events is None) or (i < max_num_events))):

            # take a step in the simulation
            time_to_next_event, before_state, next_state = self.next_event(current_state)
            current_time += time_to_next_event

            # update the state
            delta_state = np.zeros_like(self.pi_0)
            delta_state[before_state] = -1
            delta_state[next_state] = 1
            current_state += delta_state

            # only record the event, if max_temporal_resolution is exceeded
            time_since_last_event += time_to_next_event
            if time_since_last_event > max_temporal_resolution:
                time_since_last_event = 0
                i += 1

            # make room for more simulation
            if max_length <= i:
                max_length *= 2
                times = np.concatenate((times, np.zeros_like(times)), axis=0)
                state_trajectory = np.concatenate((state_trajectory, np.zeros_like(state_trajectory)), axis=0)

            # record the event
            times[i] = current_time
            state_trajectory[i] = current_state.copy()

            if return_transition_count:
                transition_count[before_state, next_state] += 1

            pbar.update(tqdm_update(delta_state))
            num_events_counter += 1

        pbar.close()

        # no reason to keep empty arrays
        times = times[:i+1]
        state_trajectory = state_trajectory[:i+1]

        if rejection_sample_num_min_events is not None:

            # handle rejection sampling pbar
            if self._rejection_sampling_pbar is None:
                self._rejection_sampling_pbar = tqdm.tqdm(desc="Rejection sampling", total=rejection_sample_num_max_tries, leave=False)
            self._rejection_sampling_pbar.update(1)
            self._rejection_sampling_pbar.set_postfix({'num_events': num_events_counter})

            if num_events_counter < rejection_sample_num_min_events:

                # if we rejection sample, we need to keep simulating again
                return self.simulate(
                    max_num_events=max_num_events,
                    continue_simulation=continue_simulation,
                    tqdm_update=tqdm_update,
                    tqdm_total=tqdm_total,
                    return_transition_count=return_transition_count,
                    max_temporal_resolution=max_temporal_resolution,
                    rejection_sample_num_min_events=rejection_sample_num_min_events,
                    rejection_sample_num_max_tries=rejection_sample_num_max_tries - 1) # decrease the number of tries by one

            # close the rejection sampling pbar upon successfull sample
            self._rejection_sampling_pbar.close()
            self._rejection_sampling_pbar = None

            # if the first state got lost, we need to add it back
            if times[0] != 0:
                times = np.concatenate(([0], times))
                state_trajectory = np.concatenate((self.pi_0[None, :], state_trajectory), axis=0)

        if return_transition_count:
            return times, state_trajectory, transition_count

        return times, state_trajectory

    def continue_simulation(
        self,
        times: np.ndarray,
        state_trajectory: np.ndarray,
        *,
        return_transition_count: bool = False,
        max_num_events: int | None = None,
        tqdm_total: int | None = None,
        continue_simulation: Callable[[np.ndarray], bool] | None = None,
        max_temporal_resolution: float = 0.,
    ):
        """
        Does what is says
        """

        times = times.copy()
        state_trajectory = state_trajectory.copy()

        if return_transition_count:
            num_states = len(self.pi_0)
            transition_count = np.zeros((num_states, num_states), dtype=int)
        
        if max_temporal_resolution:
            min_diff = np.diff(times).min()
            assert min_diff >= max_temporal_resolution, f"min_diff={min_diff} < max_temporal_resolution={max_temporal_resolution}"

        if max_num_events is not None:
            pbar = tqdm.tqdm(desc="Simulating...", total=max_num_events, leave=False)
            tqdm_update = lambda _: 1
        else:
            pbar = tqdm.tqdm(desc="Simulating...", total=tqdm_total, leave=False)

        i = len(times) - 1
        time_since_last_event = 0
        max_length = len(times)
        current_time = times[-1]
        current_state = state_trajectory[-1].copy()
        while (((continue_simulation is None) or continue_simulation(current_state)) and
               ((max_num_events is None) or (i < max_num_events))):

            # take a step in the simulation
            time_to_next_event, before_state, next_state = self.next_event(current_state)
            current_time += time_to_next_event

            # update the state
            delta_state = np.zeros_like(self.pi_0)
            delta_state[before_state] = -1
            delta_state[next_state] = 1
            current_state += delta_state

            # only record the event, if max_temporal_resolution is exceeded
            time_since_last_event += time_to_next_event
            if time_since_last_event > max_temporal_resolution:
                time_since_last_event = 0
                i += 1

            # make room for more simulation
            if max_length <= i:
                max_length *= 2
                times = np.concatenate((times, np.zeros_like(times)), axis=0)
                state_trajectory = np.concatenate((state_trajectory, np.zeros_like(state_trajectory)), axis=0)

            # record the event
            times[i] = current_time
            state_trajectory[i] = current_state.copy()

            if return_transition_count:
                transition_count[before_state, next_state] += 1

            pbar.update(tqdm_update(delta_state))

        times = times[:i+1]
        state_trajectory = state_trajectory[:i+1]

        pbar.close()
        if return_transition_count:
            return times, state_trajectory, transition_count

        return times, state_trajectory
