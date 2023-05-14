from abc import ABC, abstractmethod
from scipy import sparse
from scipy.sparse import linalg
import numpy as np

class Circuit(ABC):
    def __init__(self, basis, n, cutoff, N = None) :
        if basis not in ["charge", "flux"]:
            raise Exception('basis must be either "charge" or "flux"')

        # set basis
        self.basis = basis
        
        # initialize diagonal
        self.diag, self.delta = self.initialize_diagonal(cutoff, N)
        self.N = len(self.diag)

        # calculate hamiltonian
        self.T, self.V, self.H = self.get_T_V_H()

        # compute eigenvalues and eigenvectors
        self.E, self.psi = self.solve_circuit(n)
        self.psi = np.real(self.psi)

        # anharmonicity
        self.alpha = self.E[2] - 2 * self.E[1] + self.E[0]

    @abstractmethod
    def create_T(self):
        pass
    
    @abstractmethod
    def create_V(self):
        pass

    def initialize_diagonal(self, cutoff, N = None):
        if self.basis == "charge":
            N = int(2 * cutoff + 1)
        return np.linspace(-cutoff, cutoff, N, retstep = True)

    def get_T_V_H(self):
        T = self.create_T()
        V = self.create_V()
        return T, V, T + V

    def solve_circuit(self, n):
        if n < 3:
            raise Exception("number of eigenvalues should be at least 3")
        E, psi = sparse.linalg.eigs(self.H, k = n, which = "SR")
        E = np.real(E)
        psi *= 1 / np.sqrt(self.delta)
        return E, psi

    # matrices 
    def basis_mat(self):
        return sparse.diags(self.diag, 0, 
            shape = (self.N, self.N), dtype = np.float64)

    def create_cos_mat(self):
        if self.basis == "charge":
            return self.charge_cos_mat()
        return self.flux_cos_mat()

    def create_nn_mat(self):
        if self.basis == "charge":
            return self.charge_nn_mat()
        return self.flux_nn_mat()
    
    def identity_mat(self):
        return sparse.diags(np.ones(self.N), 0,
            shape = (self.N, self.N), dtype = np.float64)

    # flux basis matrices
    def flux_cos_mat(self):
        return sparse.diags(np.cos(self.diag), 0,
            shape = (self.N, self.N), dtype = np.float64)

    def flux_n_mat(self):
        a = np.ones(self.N) * 1j / (2 * self.delta)
        return sparse.diags(
            [a, -a, a, -a], 
            [self.N - 1, 1, -1, -self.N + 1], 
            shape = (self.N, self.N), dtype = np.cdouble)

    def flux_nn_mat(self):
        a = -1 * np.ones(self.N) / (self.delta ** 2)
        return sparse.diags(
            [a, a, -2 * a, a, a], 
            [self.N - 1, 1, 0, -1, -self.N + 1], 
            shape = (self.N, self.N), dtype = np.float64)
    
    # charge basis matrices
    def charge_nn_mat(self):
        n = self.basis_mat()
        return n @ n
    
    def charge_cos_mat(self):
        a = np.ones(self.N) / 2
        return sparse.diags(
            [a, a], [1, -1], shape = (self.N, self.N), dtype = np.float64)
