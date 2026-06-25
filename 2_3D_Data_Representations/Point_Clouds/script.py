import numpy as np
import matplotlib.pyplot as plt

theta = np.random.uniform(0, 2*np.pi, 500)
phi = np.random.uniform(0, np.pi, 500)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', s=10)
ax.set_title("3D Point Cloud")
plt.savefig("output.png")
