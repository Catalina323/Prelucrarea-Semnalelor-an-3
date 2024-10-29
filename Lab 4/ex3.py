import numpy as np
import matplotlib.pyplot as plt

def sig_sin(t, f):
    return np.sin(2 * np.pi * f * t)


fs_ok = 500
f_sig_in = 10
fs_Nyquist = 2 * f_sig_in
time_Nyquist = np.linspace(0, 1, fs_Nyquist)
time = np.linspace(0, 1, fs_ok)
points = sig_sin(time_Nyquist, f_sig_in)
sig1 = sig_sin(time, f_sig_in)
sig2 = sig_sin(time, 1)
sig3 = sig_sin(time, 19)

fig, ax = plt.subplots(4)
ax[0].plot(time, sig1)
ax[1].plot(time, sig1)
ax[1].scatter(time_Nyquist, points, color="red")
ax[2].plot(time, sig2)
ax[2].scatter(time_Nyquist, points, color="red")
ax[3].plot(time, sig3)
ax[3].scatter(time_Nyquist, points, color="red")
ax[0].set_title("semnal initial cu frecventa f = 10")
ax[1].set_title("semnal esantionat cu fs = 2 * f")
ax[2].set_title("semnal cu frecventa f = 1")
ax[3].set_title("semnal cu frecventa f = 19")
plt.tight_layout(pad=0.5)
plt.savefig("ex3.pdf")
plt.savefig("ex3.png")
plt.show()