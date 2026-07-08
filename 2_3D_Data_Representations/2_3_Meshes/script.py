import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0.5,0.5,1]])
faces = [[vertices[0], vertices[1], vertices[4]],
         [vertices[1], vertices[2], vertices[4]],
         [vertices[2], vertices[3], vertices[4]],
         [vertices[3], vertices[0], vertices[4]],
         [vertices[0], vertices[1], vertices[2], vertices[3]]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
ax.set_title("3D Mesh Representation")
plt.savefig("output.png")
