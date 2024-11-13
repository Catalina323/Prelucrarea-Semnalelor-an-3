import numpy as np

# polinoamele sunt reprezentate prin coeficienti
N = 100
g1 = np.random.randint(10)
p = np.array(np.random.randint(10, size=g1))
g2 = np.random.randint(10)
q = np.array(np.random.randint(10, size=g2))

print("p = ", p)
print("q = ", q)

prod_conv = np.polymul(p, q)
print("Inmultirea prin convolutie: ", prod_conv)

n = len(p) + len(q) - 1
n2 = 1 << (n - 1).bit_length()
p_extins = np.concatenate([p, np.array([0 for i in range(n2 - len(p))])])
q_extins = np.concatenate([q, np.array([0 for i in range(n2 - len(q))])])
p_fft = np.fft.fft(p_extins)
q_fft = np.fft.fft(q_extins)
prod = p_fft * q_fft
prod_ifft = np.fft.ifft(prod).real
prod_ifft = prod_ifft[:n]
print("Inmultirea folosind fft: ", prod_ifft)
