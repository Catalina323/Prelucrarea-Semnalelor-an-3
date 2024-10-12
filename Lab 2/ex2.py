import numpy as np
import matplotlib.pyplot as plt
import numpy.random


def semnal_sin(t, A, fi, f0):
    return A * np.sin(2 * np.pi * t * f0 + fi)

time = np.arange(0, 0.03, 0.00005)
f0 = 200
fi0 = 0
fi1 = np.pi/8
fi2 = np.pi/4
fi3 = np.pi/2
A = 1
sin_vals0 = semnal_sin(time, A, fi0, f0)
sin_vals1 = semnal_sin(time, A, fi1, f0)
sin_vals2 = semnal_sin(time, A, fi2, f0)
sin_vals3 = semnal_sin(time, A, fi3, f0)

z = numpy.random.normal(size=len(sin_vals0))
gama0 = np.sqrt(np.linalg.norm(sin_vals0, ord=2) ** 2 / (0.1 * np.linalg.norm(z) ** 2))
gama1 = np.sqrt(np.linalg.norm(sin_vals0, ord=2) ** 2 / (1 * np.linalg.norm(z) ** 2))
gama2 = np.sqrt(np.linalg.norm(sin_vals0, ord=2) ** 2 / (10 * np.linalg.norm(z) ** 2))
gama3 = np.sqrt(np.linalg.norm(sin_vals0, ord=2) ** 2 / (100 * np.linalg.norm(z) ** 2))

plt.plot(time, sin_vals0)
plt.plot(time, sin_vals1)
plt.plot(time, sin_vals2)
plt.plot(time, sin_vals3)
plt.savefig("ex2.pdf")
plt.show()

fig, a = plt.subplots(5)
a[0].plot(time, sin_vals0)
a[1].plot(time, sin_vals0 + gama0 * z)
a[2].plot(time, sin_vals0 + gama1 * z)
a[3].plot(time, sin_vals0 + gama2 * z)
a[4].plot(time, sin_vals0 + gama3 * z)
fig.savefig("ex2.2.pdf")
plt.show()
