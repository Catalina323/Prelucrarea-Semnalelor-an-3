import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# a)

N = 1000
time = np.array([i for i in range(N)])

time = np.arange(0, 1, 1/N)

trend = 4 * time * time + 12 * time + 3
sezon = np.sin(2 * np.pi * time * 100) + np.sin(2 * np.pi * time * 150)
variatii = np.random.normal(0, 0.1, N)
serie = trend + sezon + variatii


fig, ax = plt.subplots(4)
ax[0].plot(time[:100], trend[:100])
ax[0].set_title("trend")
ax[1].plot(time[:100], sezon[:100])
ax[1].set_title("sezon")
ax[2].plot(time[:100], variatii[:100])
ax[2].set_title("variatii")
ax[3].plot(time[:100], serie[:100])
ax[3].set_title("seria finala")
plt.tight_layout(pad=0.5)
plt.show()

# b)

cor = np.correlate(serie, serie, "full") / N

c1 = np.array([np.sum(serie[:k] * serie[N-k:]) / N for k in range(N)])
c2 = np.array([np.sum(serie[k:] * serie[:N-k]) / N for k in range(1, N)])
cor2 = np.concatenate((c1, c2))

y = np.arange(1999) - 1000
fig, ax = plt.subplots(2)
ax[0].plot(y, cor)
ax[0].set_title("corelatia cu functia din numpy")
ax[0].set_ylabel("Corelatie")
ax[0].set_xlabel("Shiftare")
ax[1].plot(y, cor2)
ax[1].set_ylabel("Corelatie")
ax[1].set_xlabel("Shiftare")
ax[1].set_title("corelatia implementata")
plt.tight_layout(pad=0.5)
plt.savefig("ex1b.pdf")
plt.savefig("ex1b.png")
plt.show()

# d)

p_opt = 0
m_opt = 0
min_err = float('inf')
ps = []
for p in range(1, 90):
    for m in range(100, 900, 2):
        y = np.transpose(np.flip(serie[:m + p]))

        Y = np.array([y[i:p + i] for i in range(1, m)])
        Y_t = np.transpose(Y)
        Y_c = np.matmul(np.linalg.inv(np.matmul(Y_t, Y)), Y_t)

        x = np.matmul(Y_c, y[:m - 1])

        pred = np.matmul(np.transpose(x), y[:p])
        gt = serie[m + p]

        err = mean_squared_error([pred], [gt])

        if err < min_err:
            p_opt = p
            m_opt = m
            min_err = err

print(f"p optim={p_opt}, m optim={m_opt}")

# c)

# p = 12
# m = 248
p = p_opt
m = m_opt
nr_predictii = 25

y = np.transpose(np.flip(serie[:m + p]))

for j in range(nr_predictii):
    Y = np.array([y[i:p + i] for i in range(1, m)])

    Y_t = np.transpose(Y)
    Y_c = np.matmul(np.linalg.inv(np.matmul(Y_t, Y)), Y_t)
    x = np.matmul(Y_c, y[:m - 1])

    pred = np.matmul(np.transpose(x), y[:p])
    plt.stem(time[m + j + p - 1], pred, linefmt='g-', markerfmt='go')
    plt.stem(time[m + j + p - 1], serie[m + j + p], linefmt='r-', markerfmt='ro')
    pred = np.array([pred])
    y = np.concatenate((pred, y))

plt.plot(time[m - 40:m + p - 1], serie[m - 40:m + p - 1])
plt.legend(["serie", "predictie", "adevar"])
plt.title(f"Predictia urmatoarelor {nr_predictii} valori cu p={p} si m={m}")
plt.savefig("ex1c.pdf")
plt.savefig("ex1c.png")
plt.show()
