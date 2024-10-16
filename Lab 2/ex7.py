import numpy as np
import matplotlib.pyplot as plt

def semnal_sin(t):
    return np.sin(2 * np.pi * t * 240)

def decimare(v):
    return np.array([v[i] for i in range(len(v)) if i % 4 == 0])


def decimare2(v):
    return np.array([v[i] for i in range(1, len(v)) if (i - 1) % 4 == 0])

fs = 1000
T = 1 / fs
time = np.arange(0, 0.1, T)
sig = semnal_sin(time)
time2 = decimare(time)
sig2 = decimare(sig)
sig3 = decimare2(sig)
time3 = decimare2(time)

fig, ax = plt.subplots(4)
ax[0].plot(time, sig)
ax[0].stem(time2, sig2)
ax[1].plot(time2, sig2)
ax[1].stem(time2, sig2)
ax[2].plot(time, sig)
ax[2].stem(time3, sig3)
ax[3].plot(time3, sig3)
ax[3].stem(time3, sig3)
fig.savefig("ex7.pdf")
plt.show()

# esantioanele decimate au o forma sinusoidala de frecventa 1/4 din frecventa semnalului initial
# decimarea incepand cu al doilea element este ademanatoare cu cea incepand de la primul cu diferenta ca
# valorile sunt shiftate

