import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
import math as m
import pickle
import os

def sig_sin(t):
    return np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 30 * t)

def DFT(N, sig):
    F = []
    for a in range(N):
        line = []
        for b in range(N):
            nr = m.e ** (-2 * m.pi * 1j * a * b / N)
            line.append(nr)
        F.append(line)
    X = np.matmul(F, sig)
    return X

Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
# Ns = [128, 256, 512, 1024, 2048, 4096]

path = "results.pkl"
if os.path.exists(path):
    try:
        with open(path, "rb") as f:
            result = pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print("Eroare")
else:
    result = []
    for N in Ns:
        print("N = ", N)
        time = np.linspace(0, 1, N)
        sig = sig_sin(time)

        t1 = perf_counter()
        DFT(N, sig)
        t2 = perf_counter()
        durata1 = t2 - t1

        t1 = perf_counter()
        np.fft.fft(sig)
        t2 = perf_counter()
        durata2 = t2 - t1
        result.append((durata1, durata2))

    with open("results.pkl", "wb") as f:
        pickle.dump(result, f)


DFT_results = [result[i][0] for i in range(7)]
FFT_results = [result[i][1] for i in range(7)]
categ = ["128", "256", "512", "1024", "2048", "4096", "8192"]

x = np.arange(len(categ))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, DFT_results, width, label='DFT', color='skyblue')
plt.bar(x + width/2, FFT_results, width, label='FFT', color='salmon')
plt.xticks(x, categ)
plt.legend()
plt.yscale("log")
plt.savefig("ex1.pdf")
plt.savefig("ex1.png")
plt.show()
