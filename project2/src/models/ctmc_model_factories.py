from typing import Callable
import numpy as np

"""
Read about the different models here
https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

Note that for each model we calculate the rate, which is proportional with the
proportion in that category. We then have to multiple the rate with the state size
as each person has the mentioned rate. For ease of computation, the Q matrix
calculations has been simplified
"""


def Q_SIR(beta: float, gamma: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generate the transition matrix for the SIR model
    susceptible-infected-recovered

    Parameters
    ----------
    beta: float
        The transmission rate
    gamma: float
        The recovery rate

    Returns
    -------
    np.ndarray
        The transition matrix
    """
    def Q(current_state: np.ndarray) -> np.ndarray:
        S, I, R = current_state
        population = current_state.sum()

        dS = -beta * I * S / population
        dR = gamma * I
        return np.array([
            [ dS , -dS ,  0 ],
            [  0 , -dR , dR ],
            [  0 ,   0 ,  0 ]])
    return Q


def Q_SIS(beta: float, gamma: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generate the transition matrix for the SIS model
    susceptible-infected-susceptible-infected
    """
    def Q(current_state: np.ndarray) -> np.ndarray:
        S, I = current_state
        population = current_state.sum()

        dR = -beta * S * I / population
        dI = -gamma * I
        return np.array([
            [  dR , -dR ],
            [ -dI ,  dI ]])
    return Q


def Q_SIRVD(
    alpha: float,
    nu: float,
    mu: float,
    psi: float
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generate the transition matrix for the SIRVD model
    susceptible-infected-recovered-vaccinated-deceased
    a, v, mu, psi are the infection, vaccination, recovery, and fatality rates, respectively
    """
    def Q(current_state: np.ndarray) -> np.ndarray:
        S, I, R, V, D = current_state
        population = current_state.sum()

        dS = alpha * S * I / population
        dR = mu * I
        dV = nu * S
        dD = psi * I
        return np.array([
            [ -(dS + dV) ,         dS ,  0 , dV ,  0 ],
            [          0 , -(dR + dD) , dR ,  0 , dD ],
            [          0 ,          0 ,  0 ,  0 ,  0 ],
            [          0 ,          0 ,  0 ,  0 ,  0 ],
            [          0 ,          0 ,  0 ,  0 ,  0 ]])
    return Q


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.models.CTMC import CTMC

    alpha, nu, mu, psi = 0.3, 0.0005, 0.03, 0.07
    population = 1000
    initial_infected = 1


    get_Q = Q_SIRVD(
        alpha=alpha,
        nu=nu,
        mu=mu,
        psi=psi,)

    pi_0 = np.zeros(5, dtype=int)
    pi_0[1] = initial_infected
    pi_0[0] = population - pi_0.sum()
    
    model = CTMC(get_Q, pi_0)
    times, state_trajectory = model.simulate(
        continue_simulation = lambda state: state[1],
        tqdm_update = lambda delta_state: delta_state[2:].sum(),
        tqdm_total = population - initial_infected
    )

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
    plt.show()
