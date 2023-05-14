from circuit import Circuit
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

class Fluxonium(Circuit):
    def __init__(self, E_c, E_j, E_l, phi_external, basis, n, cutoff, N = None, periodic = False):
        self.E_c = E_c
        self.E_j = E_j
        self.E_l = E_l
        self.phi_external = phi_external

        Circuit.__init__(self, basis, n, cutoff, N, periodic)

    def create_T(self):
        nn = self.create_nn_mat()
        T = 4 * self.E_c * nn
        return T

    def create_V(self):
        cos = self.create_cos_mat(1, self.phi_external)
        phi = self.basis_mat()
        V = -1 * self.E_j * cos + 0.5 * self.E_l * phi @ phi
        return V

    def dielectric_loss(self):
        w_01 = (self.E[1] - self.E[0]) * 2 * np.pi
        T = 20 * 1e-3
        Q = 5 * 1e6
        boltzmann = 1.380649e-23
        a = (w_01 ** 2) / (8 * self.E_c * np.pi * Q)
        b = np.abs(self.delta * np.conj(self.psi[:, 0]) @ self.basis_mat() @ self.psi[:, 1]) ** 2
        c = 1 / np.tanh(1.05457 * 1e-34 * w_01 / (2 * boltzmann * T)) + 1
        return a * b * c 

    def quasiparticle_tunneling(self):
        Delta_al = 82 * 2 * np.pi * 1e9 
        x_qp = 5 * 1e-9
        w_01 = (self.E[1] - self.E[0]) * 2 * np.pi
        a = np.abs(self.delta / 2 * np.conj(self.psi[:, 0]) @ self.basis_mat() @ self.psi[:, 1]) ** 2
        b = 16 * self.E_l * x_qp * np.sqrt(2 * Delta_al / w_01)
        return a * b 
