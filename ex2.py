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

# afisarea punctelor intermediare si a celor de control
fig, ax = plt.subplots()
ax.plot(W[:, 0], W[:, 1], 'o-', label='puncte intermediare')
ax.plot(P[:, 0], P[:, 1], 's--', label='puncte de control optime')
ax.legend()

# ilustrarea traiectoriei parametrizate
t = np.linspace(0, 3, 100)
Z = np.zeros((len(t), 2))
for i in range(len(t)):
    Z[i, :] = trajectory(P, t[i]).full().flatten()

ax.plot(Z[:, 0], Z[:, 1], '-o', label='traiectoria parametrizata')
ax.plot(W[:, 0], W[:, 1], 'o-', label='puncte intermediare')
ax.plot(P[:, 0], P[:, 1], 's--', label='puncte de control optime')
ax.legend()
plt.show()
