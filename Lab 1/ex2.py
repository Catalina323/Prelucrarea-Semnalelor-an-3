import numpy as np
import matplotlib.pyplot as plt

# ex 2

def semnal(t, f):
    return np.sin(2 * np.pi * f * t)


# a)
frecv_semnal = 400
time = np.arange(0, 0.03, 0.0005)
esant = np.arange(0, 0.03, 0.03/1600)
vals_esant = semnal(esant, frecv_semnal)
vals = semnal(time, frecv_semnal)
plt.stem(esant, vals_esant)
plt.plot(time, vals, color="red")
plt.savefig("2a.pdf")
plt.show()

# b)
time = np.arange(0, 3, 0.0005)
frecv_semnal = 800
vals = semnal(time, frecv_semnal)
plt.plot(time, vals)
plt.savefig("2b.pdf")
plt.show()

# c)
def semnal_sawtooth(t):
    frecv = 240
    T = 1 / frecv
    return np.mod(time, T) / T - 0.5

time = np.arange(0, 0.015, 0.00005)
vals = semnal_sawtooth(time)
plt.plot(time, vals)
plt.savefig("2c.pdf")
plt.show()

# d)
def semnal_square(t):
    return np.sign(np.sin(2 * np.pi * 300 * t))
time = np.arange(0, 0.006, 0.00005)
vals = semnal_square(time)
plt.plot(time, vals)
plt.savefig("2d.pdf")
plt.show()

# e)
arr = np.random.rand(128, 128)
plt.imshow(arr)
plt.savefig("2e.pdf")
plt.show()

# f)
def initializare():
    a = np.random.rand(128, 128)
    a = np.sort(a, axis=1)
    return a


arr = initializare()
plt.imshow(arr)
plt.savefig("2f.pdf")
plt.show()
