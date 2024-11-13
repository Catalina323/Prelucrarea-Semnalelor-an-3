import numpy as np
import matplotlib.pyplot as plt

def semnal(t):
    return np.sin(2 * np.pi * 100 * t)

def f_dreptunghiulara(dim, N):
    f = np.ones(N)
    f = np.concatenate([f, np.array([0 for _ in range(dim - N)])])
    return f

def f_hanning(dim, N):
    n = np.arange(N)
    f = 0.5 * (1 - np.cos(2 * np.pi * n / N))
    f = np.concatenate([f, np.array([0 for _ in range(dim - N)])])
    return f


fs = 2000
time = np.arange(0, 0.15, 1/fs)
dim = len(time)
N = 200

drept = f_dreptunghiulara(dim, N)
hanning = f_hanning(dim, N)
sig = semnal(time)


sig_drept = sig * drept
sig_hanning = sig * hanning

fig, ax = plt.subplots(3)
ax[0].plot(time, sig)
ax[0].set_title("Semnal initial")
ax[1].plot(time, sig_drept)
ax[1].set_title("Semnal prin fereastra derptunghiulara")
ax[2].plot(time, sig_hanning)
ax[2].set_title("Semnal prin fereastra Hanning")
plt.tight_layout(pad=1.0)
plt.savefig("ex3.pdf")
plt.savefig("ex3.png")
plt.show()
