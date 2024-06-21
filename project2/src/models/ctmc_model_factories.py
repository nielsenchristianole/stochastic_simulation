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
    def get_Q(current_state: np.ndarray) -> np.ndarray:
        S, I, R = current_state
        population = current_state.sum()

        dS = -beta * I * S / population
        dR = gamma * I
        return np.array([
            [ dS , -dS ,  0 ],
            [  0 , -dR , dR ],
            [  0 ,   0 ,  0 ]])
    return get_Q


def Q_SIS(beta: float, gamma: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generate the transition matrix for the SIS model
    susceptible-infected-susceptible-infected
    """
    def get_Q(current_state: np.ndarray) -> np.ndarray:
        S, I = current_state
        population = current_state.sum()

        dR = -beta * S * I / population
        dI = -gamma * I
        return np.array([
            [  dR , -dR ],
            [ -dI ,  dI ]])
    return get_Q


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
    def get_Q(current_state: np.ndarray) -> np.ndarray:
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
    return get_Q


def Q_custom(
    *,
    fertility_rate: float,
    healthy_death_rate: float,
    infection_rate: float,
    infection_death_rate: float,
    recovery_rate: float,
    re_susceptibility_rate: float,
    vaccination_rates: tuple[float, float, float],
    vaccinated_infection_rates: tuple[float, float, float],
    vaccinated_infection_death_rates: tuple[float, float, float],
    vaccinated_recovery_rates: tuple[float, float, float],
    vaccinated_re_susceptibility_rates: tuple[float, float, float]
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generate the transition matrix for a custom model.

    Parameters
    ----------
    fertility_rate: How many babies are born per non-infected person. Remeber to account for the number of people needed to make a baby
    healthy_death_rate: The rate at which non-infected people die. Constant accross susceptible, recovered, and vaccinated
    infection_rate: The rate at which susceptible people get infected
    infection_death_rate: The rate at which infected non-vaccinated people die
    recovery_rate: The rate at which infected people recover
    re_susceptibility_rate: The rate at which recovered people become susceptible again
    vaccination_rates: The rate at which susceptible people get vaccinated. There is 3 different vaccines.
    vaccinated_infection_rates: The rate at which vaccinated people get infected. Vaccinated people are less likely to get infected
    vaccinated_infection_death_rates: The rate at which vaccinated infected people die. Vaccinated people are less likely to die
    vaccinated_recovery_rates: The rate at which vaccinated infected people recover. Vaccinated people recover faster
    vaccinated_re_susceptibility_rates: The rate at which vaccinated recovered people become susceptible again. Vaccinated people are recovered for longer
    
    States:
    Unborn, ((susceptible | vaccination), infected, recovered) x4, deceased

    This model has susceptible and vaccinated people reproduce to create babies (assuming no pregnancy delays)
    When you are susceptible, you can die, get infected, or get vaccinated by one of 3 vaccines
    When infected, you can die or recover
    When recovered, you can die or become susceptible
    When vaccinated, you can die or get infected. The infection rate is lower than for the susceptible, but dependent on the vaccine
    When a vaccinated infected person recovers, they go to the vaccinated recovered state. From there they can die or go to vaccinated
    We assume complete memoryless and for all factors to be constant
    """

    vaccination_rates = np.array(vaccination_rates)
    vaccinated_infection_rates = np.array(vaccinated_infection_rates)
    vaccinated_infection_death_rates = np.array(vaccinated_infection_death_rates)
    vaccinated_recovery_rates = np.array(vaccinated_recovery_rates)
    vaccinated_re_susceptibility_rates = np.array(vaccinated_re_susceptibility_rates)
    

    def get_Q(current_state: np.ndarray) -> np.ndarray:

        # Unpack the current state
        (
            unborn,
            susceptible, infected, recovered,
            vaccinated_v1, infected_v1, recovered_v1,
            vaccinated_v2, infected_v2, recovered_v2,
            vaccinated_v3, infected_v3, recovered_v3,
            deceased
        ) = current_state

        num_infected = infected + infected_v1 + infected_v2 + infected_v3
        population = current_state[1:-1].sum() # Exclude unborn and deceased
        
        Q = np.zeros((14, 14))

        # rate of people are born
        k1 = fertility_rate * (susceptible + vaccinated_v1 + vaccinated_v2 + vaccinated_v3 + recovered + recovered_v1 + recovered_v2 + recovered_v3)
        Q[0, 0] = -k1
        Q[0, 1] = k1 # how many babies are born

        # what happens to susceptible people
        k1 = infection_rate * susceptible * num_infected / population
        k2 = healthy_death_rate * susceptible
        k3, k4, k5 = vaccination_rates * susceptible
        Q[1, 1] = -(k1 + k2 + k3 + k4 + k5)
        Q[1, 2] = k1 # how many people get infected
        Q[1, 4] = k3 # how many people get vaccinated by vaccine 1
        Q[1, 7] = k4 # how many people get vaccinated by vaccine 2
        Q[1, 10] = k5 # how many people get vaccinated by vaccine 3
        Q[1, 13] = k2 # how many people die

        # what happens to infected people
        k1 = infection_death_rate * infected
        k2 = recovery_rate * infected
        Q[2, 2] = -(k1 + k2)
        Q[2, 3] = k2 # how many people recover
        Q[2, 13] = k1 # how many people die

        # what happens to recovered people
        k1 = healthy_death_rate * recovered
        k2 = re_susceptibility_rate * recovered
        Q[3, 3] = -(k1 + k2)
        Q[3, 1] = k2
        Q[3, 13] = k1

        # what happens to vaccinated people
        vaccinated = np.array((vaccinated_v1, vaccinated_v2, vaccinated_v3))
        k1 = vaccinated_infection_rates * vaccinated * num_infected / population
        k2 = healthy_death_rate * vaccinated
        k_out = - (k1 + k2)
        Q[4, 4] = k_out[0]
        Q[7, 7] = k_out[1]
        Q[10, 10] = k_out[2]
        # how many people get infected
        Q[4, 5] = k1[0]
        Q[7, 8] = k1[1]
        Q[10, 11] = k1[2]
        # how many people die
        Q[4, 13] = k2[0]
        Q[7, 13] = k2[1]
        Q[10, 13] = k2[2]

        # what happens to vaccinated infected people
        vaccinated_infected = np.array((infected_v1, infected_v2, infected_v3))
        k1 = vaccinated_infection_death_rates * vaccinated_infected
        k2 = vaccinated_recovery_rates * vaccinated_infected
        k_out = - (k1 + k2)
        Q[5, 5] = k_out[0]
        Q[8, 8] = k_out[1]
        Q[11, 11] = k_out[2]
        # how many people recover
        Q[5, 6] = k2[0]
        Q[8, 9] = k2[1]
        Q[11, 12] = k2[2]
        # how many people die
        Q[5, 13] = k1[0]
        Q[8, 13] = k1[1]
        Q[11, 13] = k1[2]

        # what happens to vaccinated recovered people
        vaccinated_recovered = np.array((recovered_v1, recovered_v2, recovered_v3))
        k1 = healthy_death_rate * vaccinated_recovered
        k2 = vaccinated_re_susceptibility_rates * vaccinated_recovered
        k_out = - (k1 + k2)
        Q[6, 6] = k_out[0]
        Q[9, 9] = k_out[1]
        Q[12, 12] = k_out[2]
        # how many people become susceptible as vaccinated
        Q[6, 4] = k2[0]
        Q[9, 7] = k2[1]
        Q[12, 10] = k2[2]
        # how many people die
        Q[6, 13] = k1[0]
        Q[9, 13] = k1[1]
        Q[12, 13] = k1[2]

        return Q
    return get_Q
