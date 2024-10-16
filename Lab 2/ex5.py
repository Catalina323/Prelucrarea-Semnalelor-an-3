import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


def semnal_sinusoidal(t, f):
    return np.sin(2 * np.pi * t * f)


time = np.arange(0, 1, 0.00005)
sig0 = semnal_sinusoidal(time, 200)
sig1 = semnal_sinusoidal(time, 300)

sig_concatenate = np.concatenate((sig0, sig1))
time = np.arange(0, 2, 0.00005)
# plt.plot(time, sig_concatenate)
# plt.savefig("ex5.pdf")
# plt.show()

rate = int(10e5)
wav.write("ex5.wav", rate, sig_concatenate)



