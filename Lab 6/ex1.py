import numpy as np
import matplotlib.pyplot as plt


N = 100
x = np.random.rand(N)
fix, ax = plt.subplots(5)
ax[0].plot(x)
for i in range(1, 5):
    x = x * x
    ax[i].plot(x)
plt.show()
plt.savefig("ex1.pdf")
plt.savefig("ex1.png")

