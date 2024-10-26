import numpy as np
import matplotlib.pyplot as plt

def sig_sin(t, f):
    return np.sin(2 * np.pi * f * t)


fs_subNyquist = 10
time_subNyquist = np.linspace(0, 1, fs_subNyquist)
fs_ok = 500
f_sig_in = 10
time = np.linspace(0, 1, fs_ok)
points = sig_sin(time_subNyquist, f_sig_in)
sig1 = sig_sin(time, f_sig_in)
sig2 = sig_sin(time, 1)
sig3 = sig_sin(time, 19)

fig, ax = plt.subplots(4)
ax[0].plot(time, sig1)
ax[1].plot(time, sig1)
ax[1].scatter(time_subNyquist, points, color="red")
ax[2].plot(time, sig2)
ax[2].scatter(time_subNyquist, points, color="red")
ax[3].plot(time, sig3)
ax[3].scatter(time_subNyquist, points, color="red")
plt.savefig("ex2.pdf")
plt.show()