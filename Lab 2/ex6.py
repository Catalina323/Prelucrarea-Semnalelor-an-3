import numpy as np
import matplotlib.pyplot as plt

def semnal_sin(t, f):
    return np.sin(2 * np.pi * f * t)


fs = 2000
#time = np.arange(0, 0.05, 1 / fs)    # asta da bug :((((((
time = np.linspace(0, 1, fs)

time = time[0:150]

sig0 = semnal_sin(time, fs / 2)
sig1 = semnal_sin(time, fs / 4)
sig2 = semnal_sin(time, 0)

fig, ax = plt.subplots(3)
ax[0].plot(time, sig0)
ax[1].plot(time, sig1)
ax[2].plot(time, sig2)
fig.savefig("ex6.pdf")
plt.show()
