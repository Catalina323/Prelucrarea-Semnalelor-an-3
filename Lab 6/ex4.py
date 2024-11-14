import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import scipy
import pandas as pd

train_df = pd.read_csv("train.csv", parse_dates=["Datetime"], dayfirst=True)
time = train_df["ID"].values
x = train_df["Count"].values


def filtru(w, sig):
    return np.convolve(sig, np.ones(w), 'valid') / w

# o saptamana
# fs = 1 ora
# primele 3 zile
fs = 1 / 3600
x = x[:72]
time = time[:72]

# b)
sig_w5 = filtru(5, x)
sig_w7 = filtru(7, x)
sig_w13 = filtru(13, x)
sig_w17 = filtru(17, x)

plt.plot(x)
plt.plot(time[:len(sig_w5)], sig_w5, alpha=0.5, label="w = 5")
plt.plot(time[:len(sig_w7)], sig_w7, alpha=0.5, label="w = 7")
plt.plot(time[:len(sig_w13)], sig_w13, alpha=0.5, label="w = 13")
plt.plot(time[:len(sig_w17)], sig_w17, alpha=0.5, label="w = 17")
plt.legend()
plt.savefig("ex4b.pdf")
plt.savefig("ex4b.png")
plt.show()

# c)

# d)

N = 5
rp = 5
Wn = fs * 0.05

b1, a1 = scipy.signal.butter(N, Wn, btype='low', fs=fs)
x_filt_butter = scipy.signal.filtfilt(b1, a1, x)

b2, a2 = scipy.signal.cheby1(N, rp, Wn, btype='low', fs=fs)
x_filt_cheby1 = scipy.signal.filtfilt(b2, a2, x)


plt.plot(x, label="Semnal initial")
plt.plot(x_filt_butter, alpha=0.5, label="Butterworth")
plt.plot(x_filt_cheby1, alpha=0.5, label="Chebyshev")
plt.legend()
plt.savefig("ex4d.pdf")
plt.savefig("ex4d.png")
plt.suptitle("Filtrare cu parametrii: N=5 si rp=5")
plt.show()

# e)

# Eu as alege filtrul Butterworth deoarece semnalul filtrat respecta mai mult tendintele
# de urcare si coborare ale semnalului initial

# f)

fig, ax = plt.subplots(3)
rp = 5
i = 0

for N in [3, 5, 7]:
    b1, a1 = scipy.signal.butter(N, Wn, btype='low', fs=fs)
    x_filt_butter = scipy.signal.filtfilt(b1, a1, x)

    b2, a2 = scipy.signal.cheby1(N, rp, Wn, btype='low', fs=fs)
    x_filt_cheby1 = scipy.signal.filtfilt(b2, a2, x)

    ax[i].plot(x)
    ax[i].plot(x_filt_butter, alpha=0.5)
    ax[i].plot(x_filt_cheby1, alpha=0.5)
    ax[i].set_title(f"Ordinul: {N}")
    i += 1

fig.legend(["Semnal initial", "Butterworth", "Chebyshev"])
plt.tight_layout(pad=0.25)
plt.savefig("ex4f.pdf")
plt,savefig("ex4f.png")
plt.show()


fig, ax = plt.subplots(3)
N = 5
i = 0

b1, a1 = scipy.signal.butter(N, Wn, btype='low', fs=fs)
x_filt_butter = scipy.signal.filtfilt(b1, a1, x)

for rp in [3, 5, 7]:

    b2, a2 = scipy.signal.cheby1(N, rp, Wn, btype='low', fs=fs)
    x_filt_cheby1 = scipy.signal.filtfilt(b2, a2, x)

    ax[i].plot(x)
    ax[i].plot(x_filt_butter, alpha=0.5)
    ax[i].plot(x_filt_cheby1, alpha=0.5)
    ax[i].set_title(f"rp: {rp}")
    i += 1

fig.legend(["Semnal initial", "Butterworth", "Chebyshev"])
plt.tight_layout(pad=0.25)
plt.savefig("ex4f_2.pdf")
plt,savefig("ex4f_2.png")
plt.show()


# Filtrarea cu parametrii optimi
N_optim = 7
rp_optim = 3

b1, a1 = scipy.signal.butter(N_optim, Wn, btype='low', fs=fs)
x_filt_butter = scipy.signal.filtfilt(b1, a1, x)

b2, a2 = scipy.signal.cheby1(N_optim, rp_optim, Wn, btype='low', fs=fs)
x_filt_cheby1 = scipy.signal.filtfilt(b2, a2, x)


plt.plot(x, label="Semnal initial")
plt.plot(x_filt_butter, alpha=0.5, label="Butterworth")
plt.plot(x_filt_cheby1, alpha=0.5, label="Chebyshev")
plt.legend()
plt.suptitle("Filtre cu parametrii optimi: N=7, rp=3")
plt.savefig("ex4f_3.pdf")
plt.savefig("ex4f_3.png")
plt.show()