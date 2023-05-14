import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

C_1 = 70 * 1e-15
C_2 = 72 * 1e-15
C_c = 200 * 1e-15
C_1c = 4 * 1e-15
C_2c = 4.2 * 1e-15
C_12 = 0.1 * 1e-15
w_1 = 4 * 1e9
w_2 = 4 * 1e9

eta = C_1c * C_2c / (C_12 * C_c)

X = np.array([[0, 1], 
              [1, 0]], dtype = np.cdouble)

Y = np.array([[0, -1j], 
              [1j, 0]], dtype = np.cdouble)

Z = np.array([[1, 0], 
              [0, -1]], dtype = np.cdouble)

I = np.array([[1, 0], 
              [0, 1]], dtype = np.cdouble)

P = 0.5 * (X + 1j * Y)
M = 0.5 * (X - 1j * Y)

ZI = np.kron(Z, I)
IZ = np.kron(I, Z)
PM_MP = np.kron(P, M) + np.kron(M, P)

w_c_off = 5426426426.426426  
w_c_on = 4.75 * 1e9
T_g = 9.753644653934188e-07

w_c = lambda t : w_c_off - (w_c_off - w_c_on) * np.sin(np.pi * t / T_g)
   
def g(t):
    Delta_1 = w_1 - w_c(t)
    Delta_2 = w_2 - w_c(t)
    Delta = 2 * Delta_1 * Delta_2 / (Delta_1 + Delta_2)

    Sigma_1 = w_1 + w_c(t)
    Sigma_2 = w_2 + w_c(t)
    Sigma = 2 * Sigma_1 * Sigma_2 / (Sigma_1 + Sigma_2)

    a = 0.5 * (w_c(t) * eta / (2 * Delta) - w_c(t) * eta / (2 * Sigma) + eta + 1)
    b = C_12 * np.sqrt(w_1 * w_2) / np.sqrt(C_1 * C_2)
    return a * b

def H(t):
    Delta_1 = w_1 - w_c(t)
    Delta_2 = w_2 - w_c(t)

    g_1 = 0.5 * C_1c / np.sqrt(C_1 * C_c) * np.sqrt(w_1 * w_c(t))
    g_2 = 0.5 * C_2c / np.sqrt(C_2 * C_c) * np.sqrt(w_2 * w_c(t))

    w_1_tilde = w_1 + g_1 * g_1 / Delta_1
    w_2_tilde = w_2 + g_2 * g_2 / Delta_2

    a = 0.5 * (w_1_tilde * ZI + w_2_tilde * IZ)
    b = g(t) * PM_MP 
    return a + b

# fidelity
F = lambda M : 1 / 20 * (np.trace(M @ np.conj(M.T)) + np.abs(np.trace(M)) ** 2)
U_desired = np.array([[1, 0, 0, 0],
                      [0, 0, -1j, 0],
                      [0, -1j, 0, 0],
                      [0, 0, 0, 1]], dtype = np.cdouble)

# time evolution
U_0 = np.kron(I, I)
def f(t, U):
    return (-1j * H(t) @ U.reshape(4, 4)).flatten()
sol = integrate.solve_ivp(f, [0, T_g], U_0.flatten(), rtol = 1e-13)
t, U = sol.t, sol.y.reshape(4, 4, -1)
U_last = U[:, :, -1]

print(f"U:")
for i in range(4):
    for y in range(4):
        print(i, y, np.round(U_last[i, y], 4))

print("\nHermitian:")
H = U_last @ np.conj(U_last.T)
for i in range(4):
    for y in range(4):
        print(i, y, np.round(H[i, y], 4))

# calculate integral
integral = np.abs(integrate.quad(g, 0, T_g, epsrel = 1e-14)[0])
print(f"\nintegral: {integral}\n")

fidelity_original = F(np.conj(U_last.T) @ U_desired)
print(f"\nfidelity original: {fidelity_original}\n")

# solve equation for rotation angles
a = np.angle(U_last[0, 0])
c = np.angle(U_last[1, 2])
d = np.angle(U_last[2, 1])
f = np.angle(U_last[3, 3])
A_eq = np.array([[1, 1, 1],
              [1, -1, -1],
              [1, 1, -1]], dtype = np.float64)
b_eq = np.array([-a / np.abs(U_last[0, 0]), -f / np.abs(U_last[3, 3]), (-np.pi / 2 - c) / np.abs(U_last[1, 2])], dtype = np.float64)
solution = np.linalg.solve(A_eq, b_eq)
phi, theta1, theta2 = solution
big_mat = np.array([
    [np.exp(1j*(phi + theta1 + theta2)), 0, 0, 0],
    [0, np.exp(1j*(phi + theta1 - theta2)), 0, 0],
    [0, 0, np.exp(1j*(phi - theta1 + theta2)), 0],
    [0, 0, 0, np.exp(1j*(phi - theta1 - theta2))]], dtype = np.cdouble)

# rotate U
U_modified = big_mat @ U_last
print(f"\nU_modified: {np.round(U_modified, 3)}\n")

fidelity_modified = F(np.conj(U_desired.T) @ U_modified)
print(f"\nFidelity_modified: {fidelity_modified}")

fig = plt.figure()
plt.title("COUPLING")
plt.plot(sol.t, g(sol.t), color = "darkblue")
plt.show()

fig = plt.figure()
plt.plot(t, w_c(t), color = "darkred")
plt.title("w_c")
plt.show()

fig = plt.figure()
y_0_00 = np.array([1.0, 0.0, 0.0, 0.0], dtype = np.cdouble)
y_0_01 = np.array([0.0, 1.0, 0.0, 0.0], dtype = np.cdouble)
y_0_10 = np.array([0.0, 0.0, 1.0, 0.0], dtype = np.cdouble)
y_0_11 = np.array([0.0, 0.0, 0.0, 1.0], dtype = np.cdouble)
y_0_list = [y_0_00, y_0_01, y_0_10, y_0_11]
y_0_real = np.array([0.0, 0.0, 1.0, 0.0], dtype = np.cdouble)
probs = np.zeros((len(sol.t), 4))
for i in range(len(sol.t)):
    for j, y_0 in enumerate(y_0_list):
        y = U[:, :, i] @ y_0_real 
        probs[i, j] = np.abs(np.vdot(y_0, y)) ** 2
plt.plot(sol.t, probs[:, 0], label ="|00>")
plt.plot(sol.t, probs[:, 1], label ="|01>")
plt.plot(sol.t, probs[:, 2], label ="|10>")
plt.plot(sol.t, probs[:, 3], label ="|11>")
plt.legend()
plt.show()
