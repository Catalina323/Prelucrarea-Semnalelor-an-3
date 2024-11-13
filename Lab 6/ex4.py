import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import alpha

x = np.genfromtxt("Train.csv", delimiter=",", skip_header=True, usecols=-1, dtype=int)

def filtru(w, sig):
    return np.convolve(sig, np.ones(w), 'valid') / w

# o saptamana
# fs = 1 ora
# primele 3 zile
x = x[100:100+24*3]

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




