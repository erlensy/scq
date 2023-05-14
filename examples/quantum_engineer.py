import sys
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import scipy.integrate as integrate
from scipy.integrate import solve_ivp

sys.path.insert(1, "../src/")
from transmon import Transmon
from fluxonium import Fluxonium

def plot_pi_pulse(t, dT, U, y_0, signal):
    y = np.zeros((len(t), 3), dtype = np.cdouble)
    for i in range(len(t)):
        y[i, :] = U[:, :, i] @ y_0

    fig = plt.figure()
    ax = plt.gca()
    plt.plot(t, signal(t), label = "$s(t)$", linestyle = "--", color = "gray")
    colors = ["blue", "red", "green"]
    for i in range(3):
        plt.plot(t, np.real(y[:, i] * np.conjugate(y[:, i])), label = f"$|<\psi_{i}|\psi(t)>|^2$", color = colors[i], linewidth = 2)
    ax.set_xlabel("$t / \Delta T$", fontsize = 15)
    ax.set_xticks([0, 5 * dT, 10 * dT])
    ax.set_xticklabels([0, 5, 10])
    ax.set_ylabel("Probability", fontsize = 15)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "$\\frac{1}{2}$", "1"])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc = "center", ncol = 5, bbox_to_anchor = (0.5, 1.07), framealpha = 1)
    plt.grid()
    plt.show()
    plt.close()

def plot_transmon_states(transmon):
    fig = plt.figure(figsize = (8, 5))
    ax1 = plt.gca()
    colors = plt.get_cmap("tab10", len(transmon.psi[0, :]))
    for i in range(len(transmon.psi[0, :])):
        ax1.plot(transmon.diag, transmon.E[i] + np.real(transmon.psi[:, i] * np.conjugate(transmon.psi[:, i])), 
            label = f"$|\psi_{{{i}}}$|$^2$", linewidth = 2, color = colors(i))
    ax1.plot(transmon.diag, np.real(transmon.V.diagonal(0)), 
            color = "gray", label = "V", linestyle = "--", linewidth = 2, alpha = 0.7)

    ax1.set_xlabel("$\Phi$", fontsize = 15)
    ax1.set_ylabel("Energy", fontsize = 15)
    ax1.set_xticks([transmon.diag[0], transmon.diag[0] / 2, 0, transmon.diag[-1] / 2, transmon.diag[-1]])
    ax1.set_xticklabels(["$-\pi$", "$-\pi/2$", "0", "$-\pi/2$" ,"$\pi$"])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_yticks(np.concatenate([[np.amin(np.real(transmon.V.diagonal(0)))], transmon.E, [np.amax(np.real(transmon.V.diagonal(0)))]]))
    ax1.grid(axis = "y")
    ax2 = ax1.twinx()
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(np.concatenate([[np.amin(np.real(transmon.V.diagonal(0)))], transmon.E, [np.amax(np.real(transmon.V.diagonal(0)))]]))
    ax2.set_yticklabels(["V$_{min}$"] + [f"E$_{i}$" for i in range(len(transmon.E))] + ["V$_{max}$"])
    ax1.legend(loc = "center", ncol = 5, bbox_to_anchor = (0.5, 1.07), framealpha = 1)
    plt.show()
    plt.close()

def plot_H_c(qubit):
    H_c = np.real(qubit.create_H_c())
    g1 = H_c[0, 1]
    g2 = H_c[1, 2]
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xticks([0, 1.0, 2.0])
    ax.set_xticklabels([1, 2, 3])
    ax.set_yticks([0, 1.0, 2.0])
    ax.set_yticklabels([1, 2, 3])
    plt.imshow(H_c, cmap = "coolwarm")
    cbar = plt.colorbar(ticks = [0, g1, g2], )
    cbar.ax.set_yticklabels(["0", "$g_1$", "$g_2$"])
    plt.show()
    plt.close()

if __name__ == "__main__":
    transmon_flux = Transmon(E_c = 0.1, E_j1 = 5.0, E_j2 = 4.0, phi_external = 0.0, n_g = 0.0, 
                            basis = "flux", n = 7, cutoff = np.pi, N = 500)
     
    plot_transmon_states(transmon_flux)

    transmon_charge = Transmon(E_c = 0.1, E_j1 = 5.0, E_j2 = 4.0, phi_external = 0.0, n_g = 0.0, 
                             basis = "charge", n = 3, cutoff = 10)

    plot_H_c(transmon_charge)


    y_0 = np.array([1.0, 0.0, 0.0], dtype = np.cdouble)
    gaussian = lambda t : np.exp(-0.5 * ((t - 5 * dT) / dT) ** 2) 
    gaussian2 = lambda t : np.exp(-0.5 * ((t - 5 * dT) / dT) ** 2)  + np.exp(-0.5 * ((t - 15 * dT) / dT) ** 2)  
    U_0 = np.matrix([[0.0, 1.0],
                     [1.0, 0.0]])
    F = lambda M : 1 / 6 * (np.trace(M @ M.H) + np.abs(np.trace(M)) ** 2)
    dT_list = np.array([3, 5, 15]) / np.abs(transmon_charge.alpha)
    for dT in dT_list:
        t, U = transmon_charge.simulate_pi_pulse(signal = gaussian, T_g = 10 * dT, t_end = 10 * dT)
        plot_pi_pulse(t, dT, U, y_0, gaussian)
        print(f"Fidelity: {F(U_0.H @ np.abs(U[:2, :2, -1]))}")

    for dT in dT_list:
        t, U = transmon_charge.simulate_pi_pulse(signal = gaussian2, T_g = 10 * dT, t_end = 20 * dT)
        plot_pi_pulse(t, dT, U, y_0, gaussian2)
        print(f"Fidelity: {F(U_0.H @ np.abs(U[:2, :2, -1]))}")
