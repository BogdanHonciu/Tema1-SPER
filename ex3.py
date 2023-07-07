import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import math

# punctele intermediare
W = np.array([(0, 0), (2, 1), (1, 3), (-1, 2)])

# momentele de timp asociate
T = np.array([0, 1, 2, 3])

# numarul de puncte de control
n = len(W) - 1

# definirea functiei Bezier
def bezier(P, t):
    B = ca.SX.zeros((n + 1, 1))
    for i in range(n + 1):
        B[i] = ca.DM(math.factorial(n) / (math.factorial(i) * math.factorial(n - i))) * t ** i * (1 - t) ** (n - i)
    return B

# calculul traiectoriei parametrizate
def trajectory(P, t):
    B = bezier(P, t)
    Z = ca.SX.zeros((2, 1))
    for i in range(n + 1):
        Z += P[i, :].reshape((2, 1)) * B[i]
    return Z


def optimize_bezier(W, T):
    # definirea variabilelor de optimizare
    P = ca.SX.sym('P', n + 1, 2)

    # definirea functiei cost
    cost = 0
    for i in range(len(T)):
        t = T[i]
        w = W[i, :]
        z = trajectory(P, t)
        cost += ca.sumsqr(z - w)

    opti = ca.Opti()
    opti.minimize(cost)

    # conditii initiale si finale pentru variabilele de optimizare
    opti.subject_to(P[0, :] == W[0, :])
    opti.subject_to(P[-1, :] == W[-1, :])

    # constrangeri
    opti.subject_to(P >= -10)
    opti.subject_to(P <= 10)

    opti.solver('ipopt')

    sol = opti.solve()

    return sol.value(P)

# obtinerea punctelor de control optime
P = optimize_bezier(W, T)

# ilustrarea traiectoriei parametrizate

t = np.linspace(0, 3, 100)
Z = np.zeros((len(t), 2))
for i in range(len(t)):
    Z[i, :] = trajectory(P, t[i]).full().flatten()



fig, ax = plt.subplots()
ax.plot(Z[:, 0], Z[:, 1], '-o', label='traiectoria parametrizata')
ax.plot(W[:, 0], W[:, 1], 'o-', label='puncte intermediare')
ax.plot(P[:, 0], P[:, 1], 's--', label='puncte de control optime')
ax.legend()
plt.show()

# calculul comenzilor uV si uφ
V = ca.SX.sym('V')
phi = ca.SX.sym('phi')

# derivarea traiectoriei parametrizate
Z_fun = ca.Function('Z_fun', [ca.vertcat(P), ca.SX.sym('t')], [trajectory(P, t)])
J = Z_fun.jacobian(1)
dZdt = ca.mtimes(J, ca.SX.vstack([ca.SX.ones((1, len(t))), ca.SX.diag(np.ones(len(t) - 1), -1)]))

# calculul comenzii de viteza
uV_fun = ca.Function('uV_fun', [ca.SX.sym('dZdt', 2, len(t))], [ca.sqrt(dZdt[0, :]**2 + dZdt[1, :]**2)])
uV = uV_fun(dZdt)

# calculul comenzii de directie
L = 1.0  # lungimea vehiculului
uPhi_fun = ca.Function('uPhi_fun', [ca.SX.sym('dZdt', 2, len(t)), V], [ca.atan2(L*dZdt[1, :], V*(dZdt[0, :]*dZdt[1, 1:] - dZdt[1, :-1]*dZdt[0, 1:]))])
uPhi = uPhi_fun(dZdt, V)

# afisarea comenzilor
fig, axs = plt.subplots()
axs[0].plot(t, uV.full().flatten())
axs[0].set_ylabel('uV')
axs[1].plot(t, uPhi.full().flatten())
axs[1].set_ylabel('uPhi')
axs[1].set_xlabel('t')
plt.show()
