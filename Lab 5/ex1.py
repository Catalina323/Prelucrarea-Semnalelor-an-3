import numpy as np
import matplotlib.pyplot as plt
import heapq

x = np.genfromtxt("Train.csv", delimiter=",", skip_header=True, usecols=-1, dtype=int)

# a) un esantion pe ora = un esantion la 3600 secunde  =>  Fs = 1/3600
Fs = 1 / 3600

# b)
nr_ore = len(x)
# esantioanele ocupa 18288 ore adica 2 ani si 32 de zile

# c)
N = len(x)
X = np.fft.fft(x)
X = abs(X/N)
X = X[:N//2]
f = Fs * np.linspace(0, N//2, N//2)/N
F_max = f[np.where(X == max(X))]
print("Frecventa maxima este: ", F_max)

# d)
plt.plot(f, X)
plt.savefig("ex1d.pdf")
plt.savefig("ex1d.png")
plt.show()

# e)
# print(x.mean())  # 138.95811461067368
# media e mult diferita de 0 asta inseamna ca este prezenta componenta continua asa ca o scadem din fiecare element
media = x.mean()
x = x - media

# f)
X = np.fft.fft(x)
X = abs(X/N)
X = X[:N//2]
Xi = list(enumerate(X))
cele_mai_mari_4 = heapq.nlargest(4, Xi, key=lambda v: v[1])
X_max = [v[0] for v in cele_mai_mari_4]
frecv_max = [f[v] for v in X_max]
print("Frecventele maxime din sir sunt: ", frecv_max)

# g)
# un esantion pe ore => 24 esantioane pe zi => 24 * 28 = 672 esantioane pe luna
# pt a incepe de luni, avem nevoie de un esantion cu indicele divizibil cu 7
n = 24 * 28
luna = x[1106: 1001 + n] + media
plt.plot(luna)
plt.show()
plt.savefig("ex1g.pdf")
plt.savefig("ex1g.png")

# h)
# cri cri

# i)
X = np.fft.fft(luna)
nr_comp_eliminate = 200
for i in range(nr_comp_eliminate + 1):
    X[n//2-i] = 0
    X[n//2+1] = 0

luna_modif = np.fft.ifft(X).real
plt.plot(luna)
plt.plot(luna_modif, alpha=0.7)
plt.savefig("ex1i.pdf")
plt.savefig("ex1i.png")
plt.show()

