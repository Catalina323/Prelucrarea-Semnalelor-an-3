import scipy.io.wavfile as wav
import scipy.signal as sig
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# rate = int(10e5)
# scipy.io.wavfile.write('nume.wav', rate, signal)

# rate, x = scipy.io.wavfile.read('nume.wav')

# fs = frecv de esantionare
# fs = 44100
# sounddevice.play(myarray, fs)

def semnal_sin(t, A, fi, f0):
    return A * np.sin(2 * np.pi * t * f0 + fi)

def semnal_cos(t, A, fi, f0):
    return A * np.cos(2 * np.pi * t * f0 + fi - np.pi / 2)


time = np.arange(0, 0.03, 0.00005)
f0 = 200
fi = 0
A = 1
sin_vals = semnal_sin(time, A, fi, f0)
cos_vals = semnal_cos(time, A, fi, f0)

fig, axs = plt.subplots(2)
axs[0].plot(time, sin_vals)
axs[1].plot(time, cos_vals)
fig.savefig("ex1.pdf")
plt.show()

