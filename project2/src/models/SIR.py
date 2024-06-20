from typing import Optional

import tqdm
import numpy as np


class SIR():
    """
    The non stochastic SIR model
    """

    def __init__(
        self,
        beta: float,
        gamma: float,
        population: int,
        initial_infected: int,
        *,
        only_whole_numbered_states: bool = False
    ) -> None:
        """
        Initialize the SIR model

        Parameters
        ----------
        beta: float
            The transmission rate
        gamma: float
            The recovery rate
        population: int
            The total population
        initial_infected: int
            The initial number of infected individuals
        only_whole_numbered_states: bool
            If the population should be descritezed at random
        """
        
        self.beta = beta
        self.gamma = gamma
        self.population = population
        self.initial_infected = initial_infected
        self.only_whole_numbered_states = only_whole_numbered_states

        self.susceptible = population - initial_infected
        self.infected = initial_infected
        self.recovered = 0

    def reset(
        self,
        *,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        population: Optional[int] = None,
        initial_infected: Optional[int] = None
    ) -> None:
        """
        reset the parameters of the model
        """
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if population is not None:
            self.population = population
        if initial_infected is not None:
            self.initial_infected = initial_infected

        self.susceptible = self.population - self.initial_infected
        self.infected = self.initial_infected
        self.recovered = 0
    
    def _step(self):
        """
        Perform one step of the SIR model
        """
        new_infected = self.beta * self.susceptible * self.infected / self.population
        new_recovered = self.gamma * self.infected

        if self.only_whole_numbered_states:
            # we want descrete values for the number of new infected and recovered individuals
            new_infected = (new_infected // 1) + (np.random.rand() < new_infected % 1)
            new_recovered = (new_recovered // 1) + (np.random.rand() < new_recovered % 1)

        self.susceptible -= new_infected
        self.infected += new_infected - new_recovered
        self.recovered += new_recovered

    def simulate(self,
        days: Optional[int]=None,
        *,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Simulate the SIR model for a given number of days or until the end
        """
        pbar = tqdm.tqdm(total=days, disable=not verbose, leave=False)
        trajectory = np.zeros((days or 1, 3))

        i = 0
        while True:

            # expand the trajectory when unbouned
            if days is None and trajectory.shape[0] <= i:
                trajectory = np.concatenate((trajectory, np.zeros_like(trajectory)), axis=0)

            # update
            trajectory[i] = (self.susceptible, self.infected, self.recovered)

            # early stopping
            if ((days is not None) and (i == days - 1)) or ((self.infected == 0) if self.only_whole_numbered_states else (self.infected < 0.1)):
                break

            i += 1
            self._step()
            pbar.update(1)
        pbar.close()
        
        if days is None:
            return trajectory[:i+1]

        trajectory[i+1:] = trajectory[i]
        return trajectory


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = SIR(beta=0.3, gamma=0.1, population=1000, initial_infected=1)
    trajectory = model.simulate()

    fig, ax = plt.subplots()
    for i, label in enumerate(['Susceptible', 'Infected', 'Recovered']):
        ax.plot(trajectory[:, i], label=label)
    ax.legend()
    ax.set_xlabel('Days')
    ax.set_ylabel('Population')
    plt.show()
