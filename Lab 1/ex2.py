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
# def semnal_sawtooth(t):
#     return np.mod(1, np.sin(2 * np.pi * 240 * t))
# 
# time = np.arange(0, 0.03, 0.0005)
# vals = semnal_sawtooth(time)
# plt.plot(time, vals)
# plt.savefig("2c.pdf")
# plt.show()

