import numpy as np
import matplotlib.pyplot as plt
import math as m

fig, ax = plt.subplots(8)
time = np.array([0, 1, 2, 3, 4, 5, 6, 7])
N = 8
F = []
for a in range(N):
    lineRe = []
    line = []
    lineIm = []
    for b in range(N):
        nr = m.e**(-2 * m.pi * 1j * a * b / N)
        line.append(nr)
        lineRe.append(nr.real)
        lineIm.append(nr.imag)
    ax[a].plot(time, lineRe)
    ax[a].plot(time, lineIm)
    F.append(line)
plt.show()
fig.savefig("ex1.pdf")
fig.savefig("ex1.png")

FH = np.transpose(np.conjugate(F))
FHF = np.matmul(FH, F)
dif = FHF - N * np.identity(N)

print(np.linalg.norm(dif))





