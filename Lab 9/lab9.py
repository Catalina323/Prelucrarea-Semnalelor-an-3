import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ex 1

N = 1000
# time = np.arange(0, 1, 1/N)
time = np.linspace(0, 1, N)
trend = 10 * time * time + 0.1 * time + 0.2
sezon = np.sin(2 * np.pi * time * 10) + np.sin(2 * np.pi * time * 80)
variatii = np.random.normal(0, 0.8, N)
x = trend + sezon + variatii


# ex 2
def creare_serie(alpha):
    s = np.zeros(N)
    s[0] = x[0]
    for t in range(1, N):
        s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]

    return s


s = creare_serie(0.5)
plt.plot(x)
plt.plot(s)
plt.legend(["serie initiala", "serie noua"])
plt.savefig("ex2.pdf")
plt.savefig("ex2.png")
plt.show()

# ex 2 gasim alpha optim

alphs = np.linspace(0, 1, 1000)
min_err = float("inf")
best_alpha = 0
for alpha in alphs:
    s = creare_serie(alpha)
    err = 0
    for t in range(N-1):
        err += (s[t] - x[t+1]) ** 2
    if err < min_err:
        min_err = err
        best_alpha = alpha

print(best_alpha)

# ex 3
q = 300
# trebuie ca q < N // 2
n_pred = 5

x_test = x[len(x) - n_pred:]
x_train = x[:len(x) - n_pred]

e = np.random.normal(0, 1, len(x))
e_test = e[len(e) - n_pred:]
e_train = x[:len(e) - n_pred]

mean = np.mean(x_train)
# y = ultimii q termeni ai seriei (cei mai recenti)

# y = [(i, x_train[i] - mean) for i in range(len(x_train)-q, len(x_train))]
y = [(i, x_train[i] - mean - e_train[i]) for i in range(len(x_train) - q, len(x_train))]

X = []
for i, _ in y:
    # X.append(np.concatenate((np.array([mean]),e_train[i-q:i])))
    X.append(e_train[i - q: i])

X = np.array(X)
y = np.array([a[1] for a in y])

# asta pt mai tarziu
ths, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

ths = np.concatenate((np.array([1]), ths))
ths = np.concatenate((ths, np.array([1])))

for j in range(n_pred):
    E = e[N - q - n_pred + j: N - n_pred + j + 1]

    E = np.concatenate((np.array([mean]), E))

    pred = np.dot(ths, E)

    plt.stem(time[N - n_pred + j], pred, linefmt='g-', markerfmt='go')
    plt.stem(time[N - n_pred + j], x[N - n_pred + j], linefmt='r-', markerfmt='ro')

plt.plot(time[N - 50: N - n_pred], x[N - 50: N - n_pred])
plt.title(f"Predictia urmatoarelor {n_pred} valori")
plt.legend(("grafic", "valoarea prezisa", "valoarea reala"))
plt.savefig("ex3.pdf")
plt.savefig("ex3.png")
plt.show()

# ex 4
p_opt = 2
q_opt = 2
model = ARIMA(x, order=(p_opt, 0, q_opt))
fitted_model = model.fit()

plt.plot(x, label='Seria de timp')
plt.plot(fitted_model.fittedvalues, label='Predictii', color='red')
plt.legend()
plt.savefig("ex4.pdf")
plt.savefig("ex4.png")
plt.show()