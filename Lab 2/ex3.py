import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sig
import sounddevice as sd

rate = int(10e4)
# scipy.io.wavfile.write('nume.wav', rate, signal)

# rate, x = scipy.io.wavfile.read('nume.wav')

# fs = frecv de esantionare
# fs = 44100
# sounddevice.play(myarray, fs)

def semnal(t, f):
    return np.sin(2 * np.pi * f * t)

# a)
frecv_semnal = 400
time = np.arange(0, 0.03, 0.00005)
sig0 = semnal(time, frecv_semnal)
wav.write('nume0.wav', rate, sig0)

# fs = 44100
# sd.play(sig0, fs)

sig_citit = wav.read("nume0.wav")

print((sig0 == sig_citit).all())
# b)
# time = np.arange(0, 3, 0.0005)
# frecv_semnal = 800
# sig1 = semnal(time, frecv_semnal)

# c)
# def semnal_sawtooth(t):
#     frecv = 240
#     T = 1 / frecv
#     return np.mod(time, T) / T - 0.5
#
# time = np.arange(0, 0.015, 0.00005)
# sig2 = semnal_sawtooth(time)

# d)
# def semnal_square(t):
#     return np.sign(np.sin(2 * np.pi * 300 * t))
# time = np.arange(0, 0.006, 0.00005)
# sig3 = semnal_square(time)


