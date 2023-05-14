import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.integrate import solve_ivp

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def g(w_c):
    eta = C_1c * C_2c / (C_12 * C_c)
    Sigma_1 = w_1 + w_c
    Sigma_2 = w_2 + w_c

    Delta_1 = w_1 - w_c
    Delta_2 = w_2 - w_c
    
    a = 0.5 * (w_c / 4 * (1 / Delta_1 + 1 / Delta_2 - 1 / Sigma_1 - 1 / Sigma_2) * eta + eta + 1) 
    b = C_12 * np.sqrt(w_1 * w_2) / np.sqrt(C_1 * C_2)
    return a * b

def g_1(w_c):
    return 0.5 * C_1c * np.sqrt(w_1 * w_c / (C_1 * C_c)) 

def g_2(w_c):
    return 0.5 * C_2c * np.sqrt(w_2 * w_c / (C_2 * C_c)) 

def H(w_c):
    Delta_1 = w_1 - w_c
    Delta_2 = w_2 - w_c
    Delta = 1 / ((1 / Delta_1 + 1 / Delta_2) / 2)
    
    w_1_t = w_1 + g_1(w_c) ** 2 / Delta_1
    w_2_t = w_2 + g_2(w_c) ** 2 / Delta_2
    a = 0.5 * (w_1_t * np.kron(Z, I) + w_2_t * np.kron(I, Z))
    b = g(w_c) * (np.kron(P, M) + np.kron(M, P))
    return a + b

def E3_a():
    w_c_list = np.linspace(4.0, 7.0, 1000) * 1e9
    two_g = 2 * g(w_c_list)
    best_w_c = np.round(w_c_list[np.where(two_g == find_nearest(two_g, 0.0))][0] / 1e9, 2)

    plt.hlines(0.0, 4.0, best_w_c, linestyle = "--", linewidth = 2, color = "black")
    plt.vlines(best_w_c, -20, 0, linestyle = "--", linewidth = 2, color = "black")
    plt.plot(w_c_list / 1e9, two_g / 1e6, color = "darkred", linewidth = 3)
    plt.plot(best_w_c, 0, "o", color = "darkred")
    plt.xlim(4.0, 7.0)
    plt.ylim(-20, 5)
    plt.xlabel("$\omega_c$ [GHz]", fontsize = 16)
    plt.ylabel("2$\\tilde{g}$ [MHz]", fontsize = 16)
    plt.xticks(np.array([4, best_w_c, 5, 6, 7]), fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.grid()
    plt.show()

def E3_b():
    for w in [5, 6, 7]:
        ham = H(w * 1e9)
        plt.imshow(np.real(ham))
        plt.show()

def E3_c():
    t = np.linspace(0, T_g, 1000)
    G_time = g_t(t)

    fig = plt.figure()
    plt.plot(t, G_time / 1e6, color = "darkgreen", linewidth = 3)
    plt.xlabel("t [s]", fontsize = 16)
    plt.ylabel("2$\\tilde{g}$ [MHz]", fontsize = 16)
    plt.xticks(np.linspace(0, T_g, 5))
    plt.yticks(np.linspace(0, np.min(G_time) / 1e6, 4))
    plt.grid()
    plt.show()
    plt.close()

    fig = plt.figure()
    plt.plot(t, w_c_t(t) / 1e9, color = "darkblue", linewidth = 3)
    plt.xlabel("t [s]", fontsize = 16)
    plt.ylabel("$\omega_c$ [GHz]", fontsize = 16)
    plt.yticks(np.linspace(np.max(w_c_t(t)) / 1e9, np.min(w_c_t(t)) / 1e9, 4))
    plt.xticks(np.linspace(0, T_g, 5))
    plt.grid()
    plt.show()
    plt.close()

def E3_d():
    U_0 = np.kron(I, I)
    H_t = lambda t : H(w_c_t(t))
    def f(t, U):
        return (-1j * H_t(t) @ U.reshape(4, 4)).flatten()
    sol = solve_ivp(f, [0, T_g], U_0.flatten())
    t, U = sol.t, sol.y.reshape(4, 4, len(sol.t))
    U_last = np.round(U[:, :, -1], 4)
    print(U_last)
    
if __name__ == "__main__":
    C_1 = 70 * 1e-15
    C_2 = 72 * 1e-15
    C_c = 200 * 1e-15
    C_1c = 4 * 1e-15
    C_2c = 4.2 * 1e-15
    C_12 = 0.1 * 1e-15
    w_1 = 4 * 1e9
    w_2 = 4 * 1e9

    X = np.array([[0, 1], [1, 0]], dtype = np.cdouble)
    Y = np.array([[0, -1j], [1j, 0]], dtype = np.cdouble)
    Z = np.array([[1, 0], [0, -1]], dtype = np.cdouble)
    I = np.array([[1, 0], [0, 1]], dtype = np.cdouble)
    P = np.array([[0.0, 1.0], [0.0, 0.0]], dtype = np.cdouble)
    M = np.array([[0.0, 0.0], [1.0, 0.0]], dtype = np.cdouble)

    w_c_list = np.linspace(4.0, 7.0, 1000) * 1e9
    two_g = 2 * g(w_c_list)
    w_c_off = w_c_list[np.where(two_g == find_nearest(two_g, 0.0))][0] 
    w_c_on = 4.75 * 1e9
    print(w_c_off)
    
    # find best T_g
    L = 1e-14
    R = 1e-1
    iters = 10
    tol = 1e-15
    T_g = None
    best_integral = 3 * np.pi / 2 + tol + 1
    while np.abs(best_integral - 3*np.pi/2) > tol:
        T_g_list = np.linspace(L, R, 1000)
        integrals = np.zeros(1000)
        for i in range(1000):
            T_g = T_g_list[i]
            w_c_t = lambda t: w_c_off - (w_c_off - w_c_on) * np.sin(np.pi * t / T_g)
            g_t = lambda t: g(w_c_t(t))
            integrals[i] = np.abs(quad(g_t, 0, T_g)[0])

        best_integral = find_nearest(integrals, 3*np.pi / 2)
        best_T_g_idx = np.where(integrals == best_integral)[0][0]
        if best_integral > np.pi/2:
            L = T_g_list[best_T_g_idx - 1]
            R = T_g_list[best_T_g_idx]
        else:
            R = T_g_list[best_T_g_idx + 1]
            L = T_g_list[best_T_g_idx]
        print(best_integral)
        T_g = T_g_list[best_T_g_idx]
   
    print(f"Best T_g : {T_g}")
    w_c_t = lambda t: w_c_off - (w_c_off - w_c_on) * np.sin(np.pi * t / T_g)
    g_t = lambda t: g(w_c_t(t))

    #E3_a()
    #E3_b()
    E3_c()
    E3_d()
