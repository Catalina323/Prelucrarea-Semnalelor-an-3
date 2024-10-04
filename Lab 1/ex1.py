import numpy as np
import matplotlib.pyplot as plt

# a)
time = np.arange(0, 0.03, 0.0005)

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

# fig, axs = plt.subplots(3)
# fig.subtitle("titlu")
# axs[0].plot(time, x_rez)
# axs[1].plot(time, y_rez)
# axs[2].plot(time, z_rez)
# fig.savefig("fig1.pdf")
# plt.show()


# c)
frecv = 200
T = 1/frecv
aux = np.array([i * T for i in range(6)])
x_es = np.array(x(aux))
y_es = np.array(y(aux))
z_es = np.array(z(aux))


fig2, axs2 = plt.subplots(3)
fig2.suptitle("titlu")
axs2[0].stem(time, x_es)
axs2[1].stem(time, y_es)
axs2[2].stem(time, z_es)
fig2.savefig("fig2.pdf")
plt.show()


