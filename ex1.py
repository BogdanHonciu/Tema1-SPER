import numpy as np
import matplotlib.pyplot as plt

# punctele de control
P = np.array([(0, 0), (1, 3), (4, 2), (3, -1), (0, 1)])

# numarul de puncte de control
n = len(P) - 1

# pasul de esantionare pentru t
dt = 0.01

# calculul functiei Bezier pentru fiecare punct de control si t
def bezier(t):
    B = np.zeros((n+1,))
    for i in range(n+1):
        B[i] = np.math.comb(n, i) * t**i * (1-t)**(n-i)
    return B

# calculul traiectoriei parametrizate
def trajectory(t):
    Z = np.zeros((2,))
    for i in range(n+1):
        Z += P[i] * bezier(t)[i]
    return Z

# esantionarea traiectoriei parametrizate si afișarea ei
t = np.arange(0, 1+dt, dt)
Z = np.zeros((len(t), 2))
for i in range(len(t)):
    Z[i] = trajectory(t[i])

plt.plot(P[:,0], P[:,1], 'ro', label='punctele de control')
plt.plot(Z[:,0], Z[:,1], 'b-', label='traiectoria parametrizată')
plt.legend()
plt.show()
