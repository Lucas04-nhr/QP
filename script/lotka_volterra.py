import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lotka_volterra(X, t, alpha, beta, delta, gamma):
    prey, predator = X
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]

def simulate_lotka_volterra(alpha, beta, delta, gamma, prey0, predator0, t):
    X0 = [prey0, predator0]
    sol = odeint(lotka_volterra, X0, t, args=(alpha, beta, delta, gamma))
    return sol

def plot_lotka_volterra(t, sol, file_loc='lotka_volterra.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(t, sol[:, 0], label='Prey')
    plt.plot(t, sol[:, 1], label='Predator')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Lotka-Volterra Model')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_loc)
    plt.show()


if __name__ == "__main__":
    alpha = 0.1  # Prey birth rate
    beta = 0.02  # Predation rate
    delta = 0.01  # Predator reproduction rate
    gamma = 0.1  # Predator death rate
    prey0 = 40  # Initial prey population
    predator0 = 9  # Initial predator population
    t = np.linspace(0, 200, 1000)  # Time points
    sol = simulate_lotka_volterra(alpha, beta, delta, gamma, prey0, predator0, t)
    plot_lotka_volterra(t, sol)
