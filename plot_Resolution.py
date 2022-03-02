import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

# 遍历SNR与快拍数
SNRs = np.array([-10, -5, 0, 5, 10, 20])
Ns = np.array([100, 200, 500, 1000, 2000, 5000])

RE_ESPRIT = np.load("RE_ESPRIT.npy")
RE_MUSIC = np.load("RE_MUSIC.npy")
RE_RARE = np.load("RE_RARE.npy")


fig = plt.figure(figsize=[6,6])
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(SNRs, Ns)

# Plot a basic wireframe.
ax.plot_wireframe(X, Y, RE_ESPRIT.T, label = "ESPRIT", color="blue")
ax.plot_wireframe(X, Y, RE_MUSIC.T, label = "MUSIC", color="green")
ax.plot_wireframe(X, Y, RE_RARE.T, label = "RARE", color="red")
print(RE_ESPRIT)
print(RE_RARE)
print(RE_MUSIC)
ax.set_xlabel('SNR(dB)')
ax.set_ylabel('N')
ax.set_zlabel('ratio of correct')

plt.show()