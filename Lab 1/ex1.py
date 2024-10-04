import numpy as np
import matplotlib.pyplot as plt

# a)
time = np.arange(0, 0.1, 0.0005)

# b)
def x(t):
    return np.cos(520 * np.pi * t + 3/np.pi)
def y(t):
    return np.cos(280 * np.pi * t - 3/np.pi)
def z(t):
    return np.cos(120 * np.pi * t + 3/np.pi)

x_rez = x(time)
y_rez = y(time)
z_rez = z(time)

fig, axs = plt.subplots(3)
fig.suptitle("semnalele x, y, z")
axs[0].plot(time, x_rez)
axs[1].plot(time, y_rez)
axs[2].plot(time, z_rez)
fig.savefig("1b.pdf")
plt.show()

# c)
frecv = 200
T = 1/frecv

esantion = np.arange(0, 0.1, T)

x_es = np.array(x(esantion))
y_es = np.array(y(esantion))
z_es = np.array(z(esantion))


fig2, axs2 = plt.subplots(3)
fig2.suptitle("semnalele x, y, z esantionate")
axs2[0].stem(esantion, x_es)
axs2[1].stem(esantion, y_es)
axs2[2].stem(esantion, z_es)

axs2[0].plot(time, x_rez)
axs2[1].plot(time, y_rez)
axs2[2].plot(time, z_rez)

fig2.savefig("1c.pdf")
plt.show()
