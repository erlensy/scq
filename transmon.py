from circuit import Circuit
import sys
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import numpy as np

class Transmon(Circuit):
    def __init__(self, E_c, E_j1, E_j2, phi_external, n_g, basis, n, cutoff, N = None):
        self.E_c = E_c
        self.E_j1 = E_j1
        self.E_j2 = E_j2
        self.phi_external = phi_external
        self.n_g = n_g

        Circuit.__init__(self, basis, n, cutoff, N)

    def create_T(self):
        nn = self.create_nn_mat()
        nn_g = self.n_g * self.n_g * self.identity_mat()
        T = 4 * self.E_c * (nn - nn_g)
        return T

    def create_V(self):
        cos = self.create_cos_mat()
        gamma = self.E_j2 / self.E_j1
        d = (gamma - 1) / (gamma + 1)
        E_j = (self.E_j1 + self.E_j2) * np.sqrt(np.cos(self.phi_external) ** 2 + (d * np.sin(self.phi_external)) ** 2 )
        V = - E_j * cos
        return V

    def create_H_q(self, n):
        if n > len(self.E):
            raise Exception("dim(H_q) cant be larger than number of calculated eigenvalues")
        H_q = np.zeros((n, n), dtype = np.cdouble)
        H_q[range(n), range(n)] = self.E[:n] - self.E[0]
        return H_q
    
    def create_H_c(self, n):
        if n != 3:
            raise Exception("H_c is only implemented for n=3")
        H_c = np.zeros((3, 3), dtype = np.cdouble)
        n = self.basis_mat()
        for i in range(3):
            for j in range(3):
                H_c[i, j] = np.conj(self.psi[:, i]).T @ n @ self.psi[:, j]
        return H_c

    def simulate_pi_pulse(self, signal, T_g, t_end):
        if self.basis != "charge":
            raise Exception("please use charge basis")

        H_q = self.create_H_q(n = 3)
        H_c = self.create_H_c(n = 3)
        g1 = H_c[0, 1]
        integral = integrate.quad(signal, 0, T_g)[0]
        A = np.pi / (g1 * integral)
        def f(t, U):
            H = H_q + A * signal(t) * np.cos((self.E[1] - self.E[0]) * t) * H_c 
            return (-1j * H @ U.reshape(3, 3)).flatten()

        U_0 = np.identity(3, dtype = np.cdouble)
        sol = solve_ivp(f, [0, t_end], U_0.flatten(), rtol = 1e-12, method = "DOP853")
        return sol.t, sol.y.reshape(3, 3, len(sol.t))

    def simulate_pi_pulse_DRAG(self, delta1, delta2, sigI, sigQ, T_g, t_end):
        """ does not work! """
        if self.basis != "charge":
            raise Exception("please use charge basis")
        
        H_q = self.create_H_q(n = 3)
        H_c = self.create_H_c(n = 3)
        g1 = H_c[0, 1]
        int1 = integrate.quad(sigI, 0, T_g)[0]
        A1 = np.pi / (g1 * int1)

        int2 = integrate.quad(sigQ, 0, T_g)[0]
        if int2 == 0:
            A2 = 0
        else:
            A2 = np.pi / (g1 * int2)

        def f(t, U):
            H = H_q + A1 * sigI(t) * np.cos(delta1(t) * t) * H_c + A2 * sigQ(t) * np.sin(delta2(t) * t) * H_c 
            return (-1j * H @ U.reshape(3, 3)).flatten()

        U_0 = np.identity(3, dtype = np.cdouble)
        sol = solve_ivp(f, [0, t_end], U_0.flatten())
        return sol.t, sol.y.reshape(3, 3, len(sol.t))

    def plot_H_c(self):
        H_c = np.real(self.create_H_c(3))
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
        plt.savefig("H_c.pdf", dpi = 600)
        plt.close()

    def plot_pi_pulse(self, t, dT, U, y_0, signal, filename):
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
        plt.savefig(filename, dpi = 600)
        plt.close()

    def plot3D(self, t, U, y_0):
        y = np.zeros((len(t), 3), dtype = np.cdouble)
        for i in range(len(t)):
            y[i, :] = U[:, :, i] @ y_0

        fig = plt.figure()
        ax = plt.axes(projection = "3d")
        for i in range(0, len(t), 10):
            y_lol = np.real(np.conj(y[i, :]) * y[i, :])
            plt.plot(y_lol[2], y_lol[1], y_lol[0], "x")
        plt.show()
    
    def plot(self, filename):
        if np.abs(np.amax(self.diag) - np.pi) > 1e-2 or self.basis == "charge":
            raise Exception("Plot only made for cutoff == pi in flux basis")

        fig = plt.figure(figsize = (8, 5))
        ax1 = plt.gca()
        colors = plt.get_cmap("tab10", len(self.psi[0, :]))
        for i in range(len(self.psi[0, :])):
            ax1.plot(self.diag, self.E[i] + np.real(self.psi[:, i] * np.conjugate(self.psi[:, i])), 
                label = f"$|\psi_{{{i}}}$|$^2$", linewidth = 2, color = colors(i))
        ax1.plot(self.diag, np.real(self.V.diagonal(0)), 
                color = "gray", label = "V", linestyle = "--", linewidth = 2, alpha = 0.7)

        ax1.set_xlabel("$\Phi$", fontsize = 15)
        ax1.set_ylabel("Energy", fontsize = 15)
        ax1.set_xticks([self.diag[0], self.diag[0] / 2, 0, self.diag[-1] / 2, self.diag[-1]])
        ax1.set_xticklabels(["$-\pi$", "$-\pi/2$", "0", "$-\pi/2$" ,"$\pi$"])
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.set_yticks(np.concatenate([[np.amin(np.real(self.V.diagonal(0)))], self.E, [np.amax(np.real(self.V.diagonal(0)))]]))
        ax1.grid(axis = "y")
        ax2 = ax1.twinx()
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticks(np.concatenate([[np.amin(np.real(self.V.diagonal(0)))], self.E, [np.amax(np.real(self.V.diagonal(0)))]]))
        ax2.set_yticklabels(["V$_{min}$"] + [f"E$_{i}$" for i in range(len(self.E))] + ["V$_{max}$"])
        ax1.legend(loc = "center", ncol = 5, bbox_to_anchor = (0.5, 1.07), framealpha = 1)
        plt.savefig(filename, dpi = 600)
        plt.close()
