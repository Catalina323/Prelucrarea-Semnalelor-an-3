import numpy as np
import matplotlib.pyplot as plt
import math


def sig_sin(t):
    return np.sin(2 * np.pi * t * 9)

def sig_compl(sig, t, omega):
    return sig * math.e**(-2 * np.pi * 1j * omega * t)

fs = 1000
time = np.linspace(0, 1, fs)

sig = sig_sin(time)
sig_com1 = sig_compl(sig, time, 1)
sig_com2 = sig_compl(sig, time, 2)
sig_com3 = sig_compl(sig, time, 5)
sig_com4 = sig_compl(sig, time, 9)

dist0 = np.sqrt(time**2 + sig**2)
dist1 = np.sqrt(sig_com1.real**2 + sig_com1.imag**2)

norm0 = plt.Normalize(dist0.min(), dist0.max())
norm1 = plt.Normalize(dist1.min(), dist1.max())

cmap = plt.get_cmap("viridis")


fig1, ax1 = plt.subplots(1, 2)
for i in range(len(time) - 1):
    ax1[0].plot(time[i:i+2], sig[i:i+2], color=cmap(norm0(dist0[i])))

for i in range(len(sig_com1.real) - 1):
    ax1[1].plot(sig_com1.real[i:i+2], sig_com1.imag[i:i+2], color=cmap(norm1(dist1[i])))

plt.show()
fig1.savefig("ex2_fig1.pdf")
fig1.savefig("ex2_fig1.png")


fig2, ax2 = plt.subplots(2, 2)

dist2 = np.sqrt(sig_com2.real**2 + sig_com2.imag**2)
dist3 = np.sqrt(sig_com3.real**2 + sig_com3.imag**2)
dist4 = np.sqrt(sig_com4.real**2 + sig_com4.imag**2)

norm2 = plt.Normalize(dist2.min(), dist2.max())
norm3 = plt.Normalize(dist3.min(), dist3.max())
norm4 = plt.Normalize(dist4.min(), dist4.max())

# ax2[0, 0].plot(sig_com1.real, sig_com1.imag)
for i in range(len(sig_com1.real) - 1):
    ax2[0, 0].plot(sig_com1.real[i:i+2], sig_com1.imag[i:i+2], color=cmap(norm1(dist1[i])))

# ax2[0, 1].plot(sig_com2.real, sig_com2.imag)
for i in range(len(sig_com2.real) - 1):
    ax2[0, 1].plot(sig_com2.real[i:i+2], sig_com2.imag[i:i+2], color=cmap(norm2(dist2[i])))

# ax2[1, 0].plot(sig_com3.real, sig_com3.imag)
for i in range(len(sig_com3.real) - 1):
    ax2[1, 0].plot(sig_com3.real[i:i+2], sig_com3.imag[i:i+2], color=cmap(norm3(dist3[i])))

# ax2[1, 1].plot(sig_com4.real, sig_com4.imag)
for i in range(len(sig_com4.real) - 1):
    ax2[1, 1].plot(sig_com4.real[i:i+2], sig_com4.imag[i:i+2], color=cmap(norm4(dist4[i])))

ax2[0, 0].set_title("omega = 1")
ax2[0, 1].set_title("omega = 2")
ax2[1, 0].set_title("omega = 5")
ax2[1, 1].set_title("omega = 9")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.tight_layout(pad=2.0)
plt.show()
fig2.savefig("ex2_fig2.pdf")
fig2.savefig("ex2_fig2.png")

