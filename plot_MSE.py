import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

# 遍历SNR与快拍数
SNRs = np.array([-15,-13, -11, -9, -7, -5, 0, 5])
Ns = np.array([100, 200, 500, 1000, 2000, 5000])

Errors_ESPRIT = np.load("E_ESPRIT.npy")
Errors_MUSIC = np.load("E_MUSIC.npy")
Errors_RARE = np.load("E_RARE.npy")


fig = plt.figure(figsize=[6,6])
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(SNRs, Ns)

# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Errors_ESPRIT.T, label = "ESPRIT", color="blue")
ax.plot_wireframe(X, Y, Errors_MUSIC.T, label = "MUSIC", color="green")
ax.plot_wireframe(X, Y, Errors_RARE.T, label = "RARE", color="red")
print(Errors_ESPRIT)
print(Errors_RARE)
print(Errors_MUSIC)
ax.set_xlabel('SNR(dB)')
ax.set_ylabel('N')
ax.set_zlabel('MSE')
# ax.set_zlim(0, 500)

plt.show()