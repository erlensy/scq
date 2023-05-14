from circuit import Circuit
from scipy import integrate
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

    def simulate_pi_pulse(self, signal, T_g, t_end):
        if self.basis != "charge":
            raise Exception("please use charge basis")

        H_q = self.create_H_q()
        H_c = self.create_H_c()
        g1 = H_c[0, 1]
        integral = integrate.quad(signal, 0, T_g)[0]
        A = np.pi / (g1 * integral)

        def f(t, U):
            H = H_q + A * signal(t) * np.cos((self.E[1] - self.E[0]) * t) * H_c 
            return (-1j * H @ U.reshape(3, 3)).flatten()

        U_0 = np.identity(3, dtype = np.cdouble)
        sol = integrate.solve_ivp(f, [0, t_end], U_0.flatten(), rtol = 1e-12, method = "DOP853")
        return sol.t, sol.y.reshape(3, 3, len(sol.t))

#def simulate_pi_pulse_DRAG(self, delta1, delta2, sigI, sigQ, T_g, t_end):
    #    """ does not work! """
    #    if self.basis != "charge":
    #        raise Exception("please use charge basis")
    #    
    #    H_q = self.create_H_q(n = 3)
    #    H_c = self.create_H_c(n = 3)
    #    g1 = H_c[0, 1]
    #    int1 = integrate.quad(sigI, 0, T_g)[0]
    #    A1 = np.pi / (g1 * int1)

    #    int2 = integrate.quad(sigQ, 0, T_g)[0]
    #    if int2 == 0:
    #        A2 = 0
    #    else:
    #        A2 = np.pi / (g1 * int2)

    #    def f(t, U):
    #        H = H_q + A1 * sigI(t) * np.cos(delta1(t) * t) * H_c + A2 * sigQ(t) * np.sin(delta2(t) * t) * H_c 
    #        return (-1j * H @ U.reshape(3, 3)).flatten()

    #    U_0 = np.identity(3, dtype = np.cdouble)
    #    sol = solve_ivp(f, [0, t_end], U_0.flatten())
    #    return sol.t, sol.y.reshape(3, 3, len(sol.t))
