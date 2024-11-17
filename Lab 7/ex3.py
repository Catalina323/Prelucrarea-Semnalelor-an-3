import cv2
import scipy
from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

def compute_snr(signal, noise):
    signal_power = np.sum(signal ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-10))


def elimina_zgomot(X_noisy):
    X_noisy = (X_noisy - np.min(X_noisy)) / (np.max(X_noisy) - np.min(X_noisy)) * 255
    Y_noisy = np.fft.fft2(X_noisy)
    Y_noisy_shift = np.fft.fftshift(Y_noisy)

    rows, cols = X.shape
    crow, ccol = rows // 2, cols // 2

    radius = 100
    # pentru radius = 30 valoarea snr este minima maxima (-7.5...) insa imaginea apare prea blurata

    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

    Y_noisy_shift_filtered = Y_noisy_shift * mask

    Y_noisy_filtered = np.fft.ifftshift(Y_noisy_shift_filtered)
    X_filtered = np.fft.ifft2(Y_noisy_filtered)
    X_filtered = np.real(X_filtered)

    X_filtered = (X_filtered - np.min(X_filtered)) / (np.max(X_filtered) - np.min(X_filtered)) * 255
    return X_filtered


def elimina_zgomot_butter(X_noisy):
    n = 2
    Wn = 0.2
    b, a = scipy.signal.butter(n, Wn, btype='low')

    X_filtered = scipy.signal.filtfilt(b, a, X_noisy)
    X_filtered = (X_filtered - np.min(X_filtered)) / (np.max(X_filtered) - np.min(X_filtered)) * 255

    # filtered_rows = np.apply_along_axis(lambda x: scipy.signal.lfilter(b, a, x), axis=1, arr=X_noisy)
    # X_filtered = np.apply_along_axis(lambda x: scipy.signal.lfilter(b, a, x), axis=0, arr=filtered_rows)

    return X_filtered


X = misc.face(gray=True)

pixel_noise = 200
generated_noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + generated_noise

X_filtered = elimina_zgomot(X_noisy)
X_filtered_Butter = elimina_zgomot_butter(X_noisy)
noise = X - X_filtered
noise_Butter = X - X_filtered_Butter

print(
    f"SNR inainte: {compute_snr(X, generated_noise)}, SNR filtru: {compute_snr(X, noise)},  SNR filtru Butter: {compute_snr(X, noise_Butter)}")

plt.imshow(X_filtered, cmap=plt.cm.gray)
plt.title("Imagine filtrata")
plt.savefig("ex3_filtered_img.pdf")
plt.savefig("ex3_filtered_img.png")
plt.show()

plt.imshow(X_filtered_Butter, cmap=plt.cm.gray)
plt.title("Imagine filtrata cu filtru Butterworth")
plt.savefig("ex3_butterworth_filtered_img.pdf")
plt.savefig("ex3_butterworth_filtered_img.png")
plt.show()

