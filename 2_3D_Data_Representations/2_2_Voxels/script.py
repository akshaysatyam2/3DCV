import numpy as np
import matplotlib.pyplot as plt

grid = np.zeros((10, 10, 10), dtype=bool)
grid[2:8, 2:8, 2:8] = True
grid[3:7, 3:7, 3:7] = False

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(grid, edgecolor='k')
ax.set_title("Voxel Grid Representation")
plt.savefig("output.png")
