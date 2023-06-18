import sys
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy import linalg
from matplotlib import ticker as tck

sys.path.insert(1, "../src/")
from transmon import Transmon

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
    plt.savefig(f"../figures/transmon_{np.round(dT, 2)}_{signal}.pdf", dpi = 600)
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
    plt.savefig("../figures/transmon_states.pdf", dpi = 600)
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
    plt.savefig("../figures/transmon_H_c.pdf", dpi = 600)
    plt.close()

def readout():

    # STRUCTURE: resonator TP qubit
    hbar = 1
    h = 2 * np.pi * hbar
    Chi = 30 * h * 1e6 # hz
    epsilon = 10 * h * 1e6 # Hz
    t = 500 * 1e-10 # s
    Omegas = np.linspace(-Chi * 2, Chi * 2, 1000)
    dim = 20

    # qubit operators and states
    Z = np.diag(np.array([1, -1], dtype = np.float64))
    I = np.diag(np.array([1, 1], dtype = np.float64))
    ket_q_0 = np.array([1, 0], dtype = np.float64)
    ket_q_1 = np.array([0, 1], dtype = np.float64)

    # resonator operators and states
    offdiag = np.sqrt(np.arange(1, dim, 1, dtype = np.float64))
    a_dagger = np.diag(offdiag, -1)
    a = np.diag(offdiag, 1)
    ket_r_0 = np.array([1] + [0] * (dim - 1), dtype = np.float64)

    # system operators and states
    N_I = np.kron(a_dagger @ a, I)
    ket_00 = np.kron(ket_r_0, ket_q_0)
    ket_10 = np.kron(ket_r_0, ket_q_1)
    ket_sup = np.kron(ket_r_0, (ket_q_0 + ket_q_1) / np.sqrt(2))
    initial_states = np.array([ket_00, ket_10, ket_sup])

    # create parts of hamiltonian
    H_a = np.kron(a_dagger @ a, I)
    H_bc = Chi / 2 * np.kron(a_dagger @ a, Z) + epsilon * np.kron(a + a_dagger, I)

    # solve time evolution
    photons = np.zeros((len(Omegas), len(initial_states)), dtype = np.float64)
    for i, Omega in enumerate(Omegas):
        H_rot = Omega * H_a + H_bc
        U = linalg.expm(-1j * H_rot * t)
        for j, initial_state in enumerate(initial_states):
            psi_t = U @ initial_state
            photons[i, j] = np.conj(psi_t) @ N_I @ psi_t
        print(len(Omegas) - i)

    # plot
    labels = ["${|0 \\rangle}_r \otimes {|0 \\rangle}_q$",
              "${|0 \\rangle}_r \otimes {|1 \\rangle}_q$",
              "${|0 \\rangle}_r \otimes \\frac{1}{\sqrt{2}}({|0 \\rangle}_q + {|1 \\rangle}_q)$"]
    colors = ["tab:purple", "tab:green", "tab:red"]
    fig = plt.figure(figsize = (10, 6))
    Omegas /= (1e9)
    plt.hlines(0, np.min(Omegas), np.max(Omegas), color = "gray")
    for i in range(len(initial_states)):
        plt.plot(Omegas, photons[:, i], label = labels[i], color = colors[i], linewidth = 3)
    plt.legend(loc = "center", ncol = 3, bbox_to_anchor = (0.5, 1.061), framealpha = 1, fontsize = 17)
    plt.xlabel("$\Omega$ [Ghz]", fontsize = 20)
    plt.ylabel("#Photons", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.xlim(np.min(Omegas), np.max(Omegas))
    plt.show()

def readout_2():
    h = 2 * np.pi
    E_j = 10 * h * 1e9
    E_c = E_j / 50
    g = 50 * h * 1e6

    Z = sparse.diags([1, -1], 0, shape = (2, 2), dtype = np.float64)
    I = sparse.diags([1, 1], 0, shape = (2, 2), dtype = np.float64)

    transmon_states = 2
    transmon = Transmon(E_c = E_c, E_j1 = E_j, E_j2 = E_j, phi_external = 0.0, n_g = 0.0,
                            basis = "charge", n = transmon_states, cutoff = 19)

    omega_r_list = np.linspace(3, 3.6, 400) * 1e10
    Xi_list = np.zeros(len(omega_r_list))
    for k, omega_r in enumerate(omega_r_list):
        g_ij_squared = np.zeros((transmon_states, transmon_states))
        Xi_ij = np.zeros((transmon_states, transmon_states))
        n = transmon.create_n_mat()
        for i in range(transmon_states):
            for j in range(transmon_states):
                inner_product = transmon.delta * (transmon.psi[:, i] @ n @ transmon.psi[:, j])
                g_ij_squared[i, j] = g ** 2 * np.abs(np.real(inner_product * np.conj(inner_product))) ** 2

                omega_ij = transmon.E[i] - transmon.E[j]
                Xi_ij[i, j] = g_ij_squared[i, j] * (2 * omega_ij / (omega_ij ** 2 - omega_r ** 2))

        Xi_list[k] = np.sum(Xi_ij[0, :] - Xi_ij[1, :])

    Delta = transmon.E[1] - transmon.E[0] - omega_r_list
    plt.plot(Delta, Xi_list, marker = "o", markersize = 1)
    plt.vlines(0, np.min(Xi_list), np.max(Xi_list), color = "red", linestyle = "--", linewidth = 1)
    plt.vlines(E_c, np.min(Xi_list), np.max(Xi_list), color = "red", linestyle = "--", linewidth = 1)
    plt.show()

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
