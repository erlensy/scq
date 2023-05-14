from abc import ABC, abstractmethod
from scipy import integrate, sparse
import numpy as np

class Circuit(ABC):
    def __init__(self, basis, n, cutoff, N = None, periodic = False) :
        if basis not in ["charge", "flux"]:
            raise Exception('basis must be either "charge" or "flux"')
        if basis == "charge" and periodic == True:
            raise Exception("charge basis should be periodic")

        # set basis
        self.basis = basis
        self.periodic = periodic
        
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
        indices = np.argsort(E)
        return E[indices], psi[:, indices]
        
    # matrices 
    def basis_mat(self):
        return sparse.diags(self.diag, 0, 
            shape = (self.N, self.N), dtype = np.float64)

    def create_cos_mat(self, x = 1, y = 0):
        if self.basis == "charge":
            return self.charge_cos_mat(x, y)
        return self.flux_cos_mat(x, y)

    def create_sin_mat(self, x = 1, y = 0):
        if self.basis == "charge":
            return self.charge_sin_mat(x, y)
        return self.flux_sin_mat(x, y)

    def create_n_mat(self):
        if self.basis == "charge":
            return self.basis_mat()
        if self.periodic:
            return self.flux_n_mat_periodic()
        return self.flux_n_mat()

    def create_nn_mat(self):
        if self.basis == "charge":
            return self.charge_nn_mat()
        if self.periodic:
            return self.flux_nn_mat_periodic()
        return self.flux_nn_mat()
    
    def identity_mat(self):
        return sparse.diags(np.ones(self.N), 0,
            shape = (self.N, self.N), dtype = np.float64)

    # flux basis matrices
    def flux_cos_mat(self, x, y):
        return sparse.diags(np.cos(self.diag * x + y), 0,
            shape = (self.N, self.N), dtype = np.float64)

    def flux_sin_mat(self, x, y):
        return sparse.diags(np.sin(self.diag * x + y), 0,
            shape = (self.N, self.N), dtype = np.float64)

    def flux_n_mat_periodic(self):
        a = np.ones(self.N) * 1j / (2 * self.delta)
        return sparse.diags(
            [a, -a, a, -a], 
            [self.N - 1, 1, -1, -self.N + 1], 
            shape = (self.N, self.N), dtype = np.cdouble)

    def flux_n_mat(self):
        a = np.ones(self.N) * 1j / (2 * self.delta)
        return sparse.diags(
            [-a, a], 
            [1, -1], 
            shape = (self.N, self.N), dtype = np.cdouble)
    
    def flux_nn_mat(self):
        a = -1 * np.ones(self.N) / (self.delta ** 2)
        return sparse.diags(
            [a, -2 * a, a], 
            [1, 0, -1], 
            shape = (self.N, self.N), dtype = np.float64)
            
    def flux_nn_mat_periodic(self):
        a = -1 * np.ones(self.N) / (self.delta ** 2)
        return sparse.diags(
            [a, a, -2 * a, a, a], 
            [self.N - 1, 1, 0, -1, -self.N + 1], 
            shape = (self.N, self.N), dtype = np.float64)
    
    # charge basis matrices
    def charge_nn_mat(self):
        n = self.basis_mat()
        return n @ n
    
    def charge_cos_mat(self, x, y):
        a = np.ones(self.N) / 2
        return sparse.diags(
            [a, a], [1, -1], shape = (self.N, self.N), dtype = np.float64)

    # other
    def create_H_q(self):
        dim = len(self.E)
        H_q = np.zeros((dim, dim), dtype = np.cdouble)
        H_q[range(dim), range(dim)] = self.E[:dim] - self.E[0]
        return H_q

    def create_H_c(self):
        dim = len(self.E)
        H_c = np.zeros((dim, dim), dtype = np.cdouble)
        n = self.create_n_mat()
        for i in range(dim):
            for j in range(dim):
                H_c[i, j] = np.conj(self.psi[:, i]) @ n @ self.psi[:, j]
        return H_c

    # time dep
    def create_signal_H(self, envelope_Q, envelope_I, w_d, T_g):
        H_q = self.create_H_q()
        H_c = self.create_H_c()
        g1 = H_c[0, 1]
        A = np.pi / g1 
        return lambda t : H_q + H_c * A * (envelope_Q(t) * np.cos(w_d * t) + envelope_I(t) * np.sin(w_d * t))
        
    def solve_H(self, H, T, rtol):
        def f(t, U):
            return (-1j * H(t) @ U.reshape(3, 3)).flatten()

        dim = len(self.E)
        U_0 = np.identity(dim, dtype = np.cdouble)
        sol = integrate.solve_ivp(f, [0, T], U_0.flatten(), rtol = rtol)
        return sol.t, sol.y.reshape(3, 3, len(sol.t))
