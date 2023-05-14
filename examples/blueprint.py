import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as tck
from scipy.constants import Boltzmann
from scipy import sparse

sys.path.insert(1, "../src/")
from fluxonium import Fluxonium

def plot_fluxonium_states(fluxonium):
    fig = plt.figure(figsize = (5, 5))
    ax1 = plt.gca()
    colors = plt.get_cmap("tab10", len(fluxonium.psi[0, :]))
    ax1.set_xlabel("$\phi / 2 \pi$", fontsize = 15)
    ax1.set_ylabel("$\omega_{i} / 2 \pi$ [GHz]", fontsize = 15)
    ax1.set_xlim(-1.25, 1.25)
    ax2 = ax1.twinx()
    ax1.set_ylim(-1, 10)
    ax1.set_yticks(np.arange(-1, 10, 1))
    ax2.set_ylim(-1, 10)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax2.set_yticks(fluxonium.E / 1e9)
    ax2.set_yticklabels([f"E$_{i}$" for i in range(len(fluxonium.E))])
    ax2.grid(axis = "y", linewidth = 2)
    ax2.plot(fluxonium.diag / (2 * np.pi), np.real(fluxonium.V.diagonal(0)) / 1e9, color = "black", 
            label = "V", linestyle = "--", linewidth = 3, alpha = 0.5)
    for i in range(len(fluxonium.psi[0, :])):
        ax2.plot(fluxonium.diag / (2 * np.pi), (fluxonium.E[i] / 1e9 + fluxonium.psi[:, i]), 
            label = f"$\left|{i}\\right\\rangle$", linewidth = 2.5, color = colors(i))
    ax2.legend(loc = "center", ncol = 5, bbox_to_anchor = (0.5, 1.07), framealpha = 1)
    ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax1.xaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.show()
    plt.close()

def plot_fluxonium_transition_energy_spectrum(energies, phi_external_list):
    colors = plt.get_cmap("tab10", 4)
    fig = plt.figure(figsize = (9.5, 6))
    plt.plot(phi_external_list / (2 * np.pi), (energies[:, 2] - energies[:, 0]) / 1e9, 
            color = colors(3), linewidth = 2.5, label = "$\left|0\\right\\rangle \\rightarrow \left|2\\right\\rangle$")
    plt.plot(phi_external_list / (2 * np.pi), (energies[:, 2] - energies[:, 1]) / 1e9, 
            color = colors(0), linewidth = 2.5, label = "$\left|1\\right\\rangle \\rightarrow \left|2\\right\\rangle$")
    plt.plot(phi_external_list / (2 * np.pi), (energies[:, 1] - energies[:, 0]) / 1e9, 
            color = colors(1), linewidth = 2.5, label = "$\left|0\\right\\rangle \\rightarrow \left|1\\right\\rangle$")
    ax = plt.gca()
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel("$\phi_{ext} / 2 \pi$", fontsize = 15)
    plt.ylabel("$\omega_{ij} / 2 \pi$ [GHz]", fontsize = 15)
    plt.ylim(0, 6)
    plt.xlim(0, 1)
    plt.legend(loc = "center", ncol = 3, bbox_to_anchor = (0.5, 1.04), framealpha = 1)
    plt.grid()
    plt.show()

def plot_fluxonium_energy_relaxation_times(E_j_list, E_l_list, storage):
    fig = plt.figure()
    ax = fig.gca()
    levels = np.array([300, 400, 500, 600, 700, 800, 900, 1000])
    c = ax.contour(E_j_list / 1e9, E_l_list / 1e9, storage.T / 1e-6, colors="black", linestyles = "-", linewidths = 1.5, levels = levels)
    ax.clabel(c, inline = 1, fontsize = 12)
    cpf = ax.contourf(E_j_list / 1e9, E_l_list / 1e9, storage.T / 1e-6, cmap = "viridis", levels = 500)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel("$E_j$ [GHz]", fontsize = 15)
    ax.set_ylabel("$E_L$ [GHz]", fontsize = 15)
    ax.set_yticks([0.4, 0.8, 1.2, 1.6])
    ax.set_xticks([2, 4, 6, 8])
    cbar = plt.colorbar(cpf) 
    cbar.ax.set_ylabel('T$_1$ [$\mu$s]', rotation = 270, labelpad=25, fontsize = 15)
    cbar.ax.set_yticks([300, 400, 500, 600, 700, 800, 900, 1000])
    cbar.ax.tick_params(labelsize=12)
    plt.savefig("T1_1.pdf")
    plt.show()
    plt.close()

def E3_b1():
    fluxonium = Fluxonium(E_c = 1e9, E_j = 4e9, E_l = 1e9, phi_external = np.pi, 
            basis = "flux", n = 4, cutoff = 4.0 * np.pi, N = 500, periodic = False)
    
    plot_fluxonium_states(fluxonium)

def E3_b2():
    N_phi = 100
    phi_external_list = np.linspace(0.0, 2 * np.pi, N_phi)
    energies = np.zeros((N_phi, 3))
    for i, phi_external in enumerate(phi_external_list):
        fluxonium = Fluxonium(E_c = 1e9, E_j = 4e9, E_l = 1e9,
            phi_external = phi_external, basis = "flux", n = 3, cutoff = 4.0 * np.pi, N = 200, periodic = False)
        energies[i, :] = fluxonium.E[:3]

    plot_fluxonium_transition_energy_spectrum(energies, phi_external_list)


def E3_c(): 
    N = 5
    E_j_list = 1e9 * np.linspace(1, 8, N)
    E_l_list = 1e9 * np.linspace(0.2, 1.6, N)
    storage = np.zeros((N, N))
    for i, E_j in enumerate(E_j_list):
        for j, E_l in enumerate(E_l_list):
            f = Fluxonium(E_c = 1e9, E_j = E_j, E_l = E_l, phi_external = np.pi, 
                    basis = "flux", n = 3, cutoff = 4.0 * np.pi, N = 500, periodic = False)
            a = f.dielectric_loss() * 2 / np.pi
            b = f.quasiparticle_tunneling() * 2 / np.pi
            storage[i, j] =  1 / (a + b) 
            print(i, j)

    plot_fluxonium_energy_relaxation_times(E_j_list, E_l_list, storage)

def boom():
    f_a = Fluxonium(E_c = 1.0 * 1e9, E_j = 4.0 * 1e9, E_l = 0.9 * 1e9, phi_external = np.pi, 
        basis = "flux", n = 4, cutoff = 4.0 * np.pi, N = 100, periodic = False) 

    f_b = Fluxonium(E_c = 1.0 * 1e9, E_j = 4.0 * 1e9, E_l = 1.0 * 1e9, phi_external = np.pi, 
        basis = "flux", n = 4, cutoff = 4.0 * np.pi, N = 100, periodic = False) 

    n_ab = sparse.kron(f_a.create_n_mat(), f_b.create_n_mat())
    phi_ab = sparse.kron(f_a.basis_mat(), f_b.basis_mat())
    HI_ab = sparse.kron(f_a.H, f_b.identity_mat())
    IH_ab = sparse.kron(f_a.identity_mat(), f_b.H)
    phiI_ab = sparse.kron(f_a.basis_mat(), f_b.identity_mat())

    J_l = 2 * 1e6
    J_c_list = 1e6 * np.linspace(0, 80, 50)
    storage = np.zeros((50, 2))
    for i, J_c in enumerate(J_c_list):
        H_coupl = J_c * n_ab- J_l * phi_ab 
        H = HI_ab + IH_ab + H_coupl

        E, psi = sparse.linalg.eigs(H, k = 4, which = "SR")
        E = np.real(E)
        indices = np.argsort(E)
        E = E[indices]
        psi = psi[:, indices]

        mu_a = f_a.delta * f_b.delta * np.abs(np.conj(psi[:, 0].T) @ phiI_ab @ psi[:, 2])
        mu_b = f_a.delta * f_b.delta * np.abs(np.conj(psi[:, 0].T) @ phiI_ab @ psi[:, 1])
        mu_phi = mu_a / mu_b

        zeta_zz = E[0] + E[3] - E[1] - E[2]
        storage[i, 0] = mu_phi
        storage[i, 1] = zeta_zz
        print(len(J_c_list) - i)

    # plot
    fig = plt.figure(figsize = (9.5, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(J_c_list / 1e6, storage[:, 0], color = "darkred", linewidth = 2.5, label = "$\mu_{\phi}$")
    ax2.plot(J_c_list / 1e6, np.abs(storage[:, 1]) / 1e3, color = "darkblue", linewidth = 2.5, label = "$|\zeta_{ZZ}|$")
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 0.12)
    ax2.set_ylim(0, 10)
    ax1.set_xlabel("$J_C$ [MHz]", fontsize = 15)
    ax1.set_ylabel("$\mu_{\phi}$", fontsize = 15)
    ax2.set_ylabel("$|\zeta| / 2 \pi$ [kHz]", rotation = 270, fontsize = 15)
    ax2.yaxis.set_label_coords(1.075, 0.5)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax1.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax2.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(tck.AutoMinorLocator())
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc = "center", ncol = 2, bbox_to_anchor = (0.5, 1.04))
    plt.show()
    

if __name__ == "__main__":
    E3_b1()
    #E3_b2()
    #E3_c()
    #boom()

