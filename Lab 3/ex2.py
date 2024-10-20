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


# fig, ax = plt.subplots(1, 2)
# ax[0].plot(time, sig)
# ax[1].plot(sig_com1.real, sig_com1.imag)
# plt.ylim(-1, 1)
# plt.show()
# fig.savefig("ex2_fig1.pdf")
# fig.savefig("ex2_fig1.png")

fig2, ax2 = plt.subplots(2, 2)
ax2[0, 0].plot(sig_com1.real, sig_com1.imag)
ax2[0, 0].set_title("omega = 1")
ax2[0, 1].plot(sig_com2.real, sig_com2.imag)
ax2[0, 1].set_title("omega = 2")
ax2[1, 0].plot(sig_com3.real, sig_com3.imag)
ax2[1, 0].set_title("omega = 5")
ax2[1, 1].plot(sig_com4.real, sig_com4.imag)
ax2[1, 1].set_title("omega = 9")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.tight_layout(pad=2.0)
plt.show()
fig2.savefig("ex2_fig2.pdf")
fig2.savefig("ex2_fig2.png")

