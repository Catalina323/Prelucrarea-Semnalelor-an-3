import numpy as np


N = 100
time = np.arange(0, 1, 1/N)
trend = 4 * time * time + 12 * time + 3
sezon = np.sin(2 * np.pi * time * 100) + np.sin(2 * np.pi * time * 150)
variatii = np.random.normal(0, 0.1, N)
serie = trend + sezon + variatii

# exercitiul 2

L = 20
K = N - L
X = []
for i in range(L):
    line = []
    for j in range(K):
        line.append(serie[i+j])
    X.append(line)

X = np.array(X)


# exercitiul 3

XXT = X @ X.T
XTX = X.T @ X

U, S, VT = np.linalg.svd(X)
lambda_XXT = np.linalg.eigvals(XXT)
lambda_XTX = np.linalg.eigvals(XTX)

singular_squared = S**2

# S = valorile singulare
# lambda_XXT = valorile proprii pentru XXT
# lambda_XTX = valorile proprii pentru XTX
# singular_squared = patratul valorilor singulare
# observam ca valorile ssingular_squared sunt asemanatoase cu lambda_XTX si lambda_XXT

