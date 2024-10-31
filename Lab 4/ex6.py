import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

rate, sig = wav.read("vocale.waptt.wav")

N = len(sig)
time = np.linspace(0, 4, N)

gr_dim = N // 100
sup = gr_dim // 4
grupuri = []
grupuri.append(sig[0:gr_dim + sup])
for i in range(1, 100):
    gr = sig[i * gr_dim - sup : (i + 1) * gr_dim + sup]
    grupuri.append(gr)

FFT_matr = []
for g in grupuri:
    FFT_matr.append(np.abs(np.fft.fft(g)))


FFT_matr.pop(-1)
FFT_matr.pop(0)
print(np.array(FFT_matr).T.shape)
plt.imshow(10 * np.log10(np.array(FFT_matr).T), aspect='auto')
plt.colorbar()
plt.savefig("ex6.pdf")
plt.savefig("ex6.png")
plt.show()



