from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

# ex 2

X = misc.face(gray=True)
Y = np.fft.fft2(X)

def compute_snr(signal, noise):
    signal_power = np.sum(signal**2)
    noise_power = np.sum(noise**2)
    return 10 * np.log10(signal_power / (noise_power + 1e-10))

freq_filt_optim = 140

Y_filtered = Y.copy()
freq_db = 20*np.log10(abs(Y))
Y_filtered[freq_db > freq_filt_optim] = 0
X_filtered = np.fft.ifft2(Y_filtered)
X_filtered = np.real(X_filtered)
X_filtered = (X_filtered - np.min(X_filtered)) / (np.max(X_filtered) - np.min(X_filtered)) * np.max(X)

noise = X - X_filtered

snr_value = compute_snr(X, noise)
print(snr_value)

plt.imshow(X_filtered, cmap=plt.cm.gray)
plt.savefig("ex2.pdf")
plt.savefig("ex2.png")
plt.show()


