import numpy as np
import matplotlib.pyplot as plt


# ex 1
n = 100

def plot_img(img, name, title):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title(title)
    plt.savefig(f"{name}.pdf")
    plt.savefig(f"{name}.png")
    plt.show()

def plot_spec(spec, name, title):
    plt.imshow(20 *np.log10(abs(spec) + 1e-14))
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"{name}.pdf")
    plt.savefig(f"{name}.png")
    plt.show()

# x(n1, n2) = sin(2 pi n1 + 3 pi n2)
img1 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        img1[i][j] = np.sin( 2 *np.pi *i + 3* np.pi * j)

Y1 = np.fft.fft2(img1)

plot_img(img1, "Img1", "Img: x(n1, n2) = sin(2 pi n1 + 3 pi n2)")
plot_spec(Y1, "Spec1", 'Spec: x(n1, n2) = sin(2 pi n1 + 3 pi n2)')

# x(n1, n2) = sin(4 pi n1) + cos(6 pi n2)
img2 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        img2[i][j] = np.sin(4 * np.pi * i) + np.cos(6 * np.pi * j)

Y2 = np.fft.fft2(img2)
plot_img(img2, "Img2", 'Img: x(n1, n2) = sin(4 pi n1) + cos(6 pi n2)')
plot_spec(Y2, 'Spec2', 'Spec: x(n1, n2) = sin(4 pi n1) + cos(6 pi n2)')

# Y(0,5)=Y(n-5, 0) = 1 else Y(m1, m2) = 0
Y3 = np.zeros((n, n))
Y3[0, 5] = 1
Y3[n - 5][0] = 1
img3 = np.fft.ifft2(Y3)
img3 = np.real(img3)

plot_img(img3, "Img3", "Img: Y(0,5)=Y(n-5, 0) = 1 else Y(m1, m2) = 0")
plot_spec(Y3, "Spec3", "Spec: Y(0,5)=Y(n-5, 0) = 1 else Y(m1, m2) = 0")

# Y(5,0)=Y(-5, 0) = 1 else Y(m1, m2) = 0
Y4 = np.zeros((n, n))
Y4[5, 0] = 1
Y4[-5][0] = 1
img4 = np.fft.ifft2(Y4)
img4 = np.real(img4)

plot_img(img4, "Img4", "Img: Y(5,0)=Y(-5, 0) = 1 else Y(m1, m2) = 0")
plot_spec(Y4, "Spec4", "Spec: Y(5,0)=Y(-5, 0) = 1 else Y(m1, m2) = 0")

# Y(5,5)=Y(n-5, n-5) = 1 else Y(m1, m2) = 0
Y5 = np.zeros((n, n))
Y5[5, 5] = 1
Y5[n - 5][n - 5] = 1
img5 = np.fft.ifft2(Y5)
img5 = np.real(img5)

plot_img(img5, "Img5", "Img: Y(5,5)=Y(n-5, n-5) = 1 else Y(m1, m2) = 0")
plot_spec(Y5, "Spec5", "Spec: Y(5,5)=Y(n-5, n-5) = 1 else Y(m1, m2) = 0")

