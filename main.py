import numpy as np
from scipy import sparse
from transmon import Transmon
from circuit import Circuit
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import scipy.integrate as integrate
from scipy.integrate import solve_ivp


if __name__ == "__main__":
    transmon_flux = Transmon(E_c = 0.1, E_j1 = 5.0, E_j2 = 4.0, phi_external = 0.0, n_g = 0.0, 
                            basis = "flux", n = 7, cutoff = np.pi, N = 500)
    
    transmon_flux.plot("transmon.pdf")

    transmon_charge = Transmon(E_c = 0.1, E_j1 = 5.0, E_j2 = 4.0, phi_external = 0.0, n_g = 0.0, 
                             basis = "charge", n = 3, cutoff = 10)
    
    transmon_charge.plot_H_c()
    

    y_0 = np.array([1.0, 0.0, 0.0], dtype = np.cdouble)
    gaussian = lambda t : np.exp(-0.5 * ((t - 5 * dT) / dT) ** 2) 
    gaussian2 = lambda t : np.exp(-0.5 * ((t - 5 * dT) / dT) ** 2)  + np.exp(-0.5 * ((t - 15 * dT) / dT) ** 2)  
    U_0 = np.matrix([[0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]])
    F = lambda M : 1 / 12 * (np.trace(M @ M.H) + np.abs(np.trace(M)) ** 2)

    # simulate pi pulse
    dT = 3 / np.abs(transmon_charge.alpha)
    t, U = transmon_charge.simulate_pi_pulse(signal = gaussian, T_g = 10 * dT, t_end = 10 * dT)
    transmon_charge.plot_pi_pulse(t, dT, U, y_0, gaussian, "gaussian_dT=3.alpha.pdf")
    print(F(U_0.H @ np.abs(U[:, :, -1])))

    dT = 5 / np.abs(transmon_charge.alpha)
    t, U = transmon_charge.simulate_pi_pulse(signal = gaussian, T_g = 10 * dT, t_end = 10 * dT)
    transmon_charge.plot_pi_pulse(t, dT, U, y_0, gaussian, "gaussian_dT=5.alpha.pdf")
    print(F(U_0.H @ np.abs(U[:2, :2, -1])))

    dT = 15 / np.abs(transmon_charge.alpha)
    t, U = transmon_charge.simulate_pi_pulse(signal = gaussian, T_g = 10 * dT, t_end = 10 * dT)
    transmon_charge.plot_pi_pulse(t, dT, U, y_0, gaussian, "gaussian_dT=15.alpha.pdf")
    print(F(U_0.H @ np.abs(U[:2, :2, -1])))

    # two gaussians
    dT = 3 / np.abs(transmon_charge.alpha)
    t, U = transmon_charge.simulate_pi_pulse(signal = gaussian2, T_g = 10 * dT, t_end = 20 * dT)
    transmon_charge.plot_pi_pulse(t, dT, U, y_0, gaussian2, "gaussian2_dT=3.alpha.pdf")

    dT = 5 / np.abs(transmon_charge.alpha)
    t, U = transmon_charge.simulate_pi_pulse(signal = gaussian2, T_g = 10 * dT, t_end = 20 * dT)
    transmon_charge.plot_pi_pulse(t, dT, U, y_0, gaussian2, "gaussian2_dT=5.alpha.pdf")

    dT = 15 / np.abs(transmon_charge.alpha)
    t, U = transmon_charge.simulate_pi_pulse(signal = gaussian2, T_g = 10 * dT, t_end = 20 * dT)
    transmon_charge.plot_pi_pulse(t, dT, U, y_0, gaussian2, "gaussian2_dT=15.alpha.pdf")

def drag_pulse_test():
    n_values = 4
    lmbda_list = np.linspace(0.1, 1, n_values)
    F_list = np.zeros(n_values)
    alpha = transmon_charge.alpha
    dT = 3 / np.abs(alpha)
    def g(t, dT):
        return np.exp(-0.5 * ((t - 5 * dT) / dT) ** 2)

    def sig_I(dT):
        return lambda t : g(t,dT)
    
    def sig_Q(dT, lmbda, alpha):
        return lambda t : -lmbda / alpha * (t - 5 * dT) / (dT ** 2) * g(t, dT)
    
    def delta1(dT, lmbda, alpha):
        return lambda t : (lmbda ** 2 - 4) * g(t, dT) * g(t, dT) / (4 * alpha)

    def delta2(dT, lmbda, alpha):
        return lambda t : alpha + 2 * (lmbda ** 2 - 4) * g(t, dT) * g(t, dT) / (4 * alpha)
    
    for i, lmbda in enumerate(lmbda_list):
        t, U = transmon_charge.simulate_pi_pulse_DRAG(delta1(dT, lmbda, alpha), delta2(dT, lmbda, alpha), sig_I(dT), sig_Q(dT, lmbda, alpha), 10 * dT, 15 * dT)
        print(i)
        M = U_0.H @ np.abs(U[:, :, -1])
        F_list[i] = F(M) 

    plt.scatter(lmbda_list, F_list, marker = "x", s = 100, color = "black")
    plt.xlabel("$\lambda$")
    plt.ylabel("$F$")
    plt.show()
