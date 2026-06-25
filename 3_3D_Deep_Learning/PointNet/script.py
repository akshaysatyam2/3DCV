import numpy as np
import matplotlib.pyplot as plt

points = np.random.rand(100, 2)
W = np.random.rand(2, 2)
features = points @ W

idx0 = np.argmax(features[:, 0])
idx1 = np.argmax(features[:, 1])

plt.figure()
plt.scatter(points[:,0], points[:,1], c='gray', alpha=0.5, label='All Points')
plt.scatter(points[idx0,0], points[idx0,1], c='r', s=100, label='Max pool contributor 1')
plt.scatter(points[idx1,0], points[idx1,1], c='b', s=100, label='Max pool contributor 2')
plt.title("PointNet: Global Feature Aggregation")
plt.legend()
plt.savefig("output.png")
