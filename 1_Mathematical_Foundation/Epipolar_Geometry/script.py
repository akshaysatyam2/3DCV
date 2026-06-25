import numpy as np
import matplotlib.pyplot as plt

pts3d = np.random.rand(20, 3) * 10
pts3d[:, 2] += 20

K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
P1 = K @ np.eye(3, 4)
P2 = K @ np.array([[1,0,0,-5], [0,1,0,0], [0,0,1,0]])

pts3d_hom = np.hstack([pts3d, np.ones((20,1))]).T
pts2d_1 = P1 @ pts3d_hom
pts2d_1 = pts2d_1[:2, :] / pts2d_1[2, :]
pts2d_2 = P2 @ pts3d_hom
pts2d_2 = pts2d_2[:2, :] / pts2d_2[2, :]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.scatter(pts2d_1[0], pts2d_1[1], c='r')
ax1.set_title("Left Camera View")
ax1.set_xlim(0, 640); ax1.set_ylim(0, 480); ax1.invert_yaxis()
ax2.scatter(pts2d_2[0], pts2d_2[1], c='b')
ax2.set_title("Right Camera View (Epipolar Shift)")
ax2.set_xlim(0, 640); ax2.set_ylim(0, 480); ax2.invert_yaxis()
plt.savefig("output.png")
