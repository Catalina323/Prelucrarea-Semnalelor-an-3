import numpy as np
import matplotlib.pyplot as plt

def semnal_sin(t, f):
    return np.sin(2 * np.pi * f * t)


def semnal_sawtooth(t, f):
    T = 1 / f
    return np.mod(time, T) / T - 0.5

time = np.arange(0, 0.03, 0.00005)
frecv = 240
sig_sawtooth = semnal_sawtooth(time, frecv)
sig_sin = semnal_sin(time, frecv)
sig_sum = sig_sin + sig_sawtooth

fig, ax = plt.subplots(3)
ax[0].plot(time, sig_sin)
ax[1].plot(time, sig_sawtooth)
ax[2].plot(time, sig_sum)
fig.savefig("ex4.pdf")
plt.show()