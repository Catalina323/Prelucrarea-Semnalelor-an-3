import numpy as np
import matplotlib.pyplot as plt
import math as m

def sig_sin(t):
    return np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 30 * t)

fs = 100
N = 10
time = np.linspace(0, 1, fs)
sig = sig_sin(time)

time2 = time[:N]
sig2 = sig_sin(time2)

# matricea Fourier pt N = 10
F = []
for a in range(N):
    line = []
    for b in range(N):
        nr = m.e**(-2 * m.pi * 1j * a * b / N)
        line.append(nr)
    F.append(line)

X = np.matmul(F, sig2)
coef = abs(X[:(N//2 + 1)])

frecv = np.array([0, 10, 20, 30, 40, 50])
fig, ax = plt.subplots(1, 2)
ax[0].plot(time, sig)
ax[1].stem(frecv, coef)
fig.savefig("ex3.pdf")
fig.savefig("ex3.png")
plt.show()
