import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import alpha
import scipy

x = np.genfromtxt("Train.csv", delimiter=",", skip_header=True, usecols=-1, dtype=int)

def filtru(w, sig):
    return np.convolve(sig, np.ones(w), 'valid') / w

# o saptamana
# fs = 1 ora
# primele 3 zile
fs = 1 / 3600
x = x[:72]

# b)
sig_w5 = filtru(5, x)
sig_w7 = filtru(7, x)
sig_w13 = filtru(13, x)
sig_w17 = filtru(17, x)

plt.plot(x)
plt.plot(sig_w5, alpha=0.5, label="w = 5")
plt.plot(sig_w7, alpha=0.5, label="w = 7")
plt.plot(sig_w13, alpha=0.5, label="w = 13")
plt.plot(sig_w17, alpha=0.5, label="w = 17")
plt.legend()
plt.savefig("ex4b.pdf")
plt.savefig("ex4b.png")
plt.show()

# c)

# d)

N = 5
rp = 5
Wn = fs * 0.05

b1, a1 = scipy.signal.butter(N, Wn, btype='low', fs=fs)
x_filt_butter = scipy.signal.filtfilt(b1, a1, x)

b2, a2 = scipy.signal.cheby1(N, rp, Wn, btype='low', fs=fs)
x_filt_cheby1 = scipy.signal.filtfilt(b2, a2, x)


plt.plot(x, label="Semnal initial")
plt.plot(x_filt_butter, alpha=0.5, label="Butterworth")
plt.plot(x_filt_cheby1, alpha=0.5, label="Chebyshev")
plt.legend()
plt.savefig("ex4d.pdf")
plt.savefig("ex4d.png")
plt.show()

# e)

# f)

