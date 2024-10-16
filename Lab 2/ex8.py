import numpy as np
import matplotlib.pyplot as plt

def pade(t):
    return (t - (7 * (t**3) / 60)) / (1 + (t**2 / 20))

time = np.arange(-np.pi/2, np.pi/2, 0.0005)

sin = np.sin(time)
pade_vals = pade(time)

fig, ax = plt.subplots(3)
ax[0].plot(time, sin)
ax[1].plot(time, time)
ax[2].plot(time, sin-time)
fig.suptitle("aproximarea sin(x) = x")
fig.savefig("ex8.pdf")
plt.show()

fig, ax = plt.subplots(3)
ax[0].plot(time, sin)
ax[1].plot(time, pade_vals)
ax[2].plot(time, sin-pade_vals)
fig.suptitle("aproximarea Pade")
fig.savefig("ex8.2.pdf")
plt.show()
