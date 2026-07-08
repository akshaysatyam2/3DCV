import numpy as np
import matplotlib.pyplot as plt

def create_rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def translate(dx, dy, dz):
    return np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])

origin = np.array([[0,0,0,1], [1,0,0,1], [0,1,0,1], [0,0,1,1]])
T = translate(2, 2, 0)
R = create_rotation_matrix_z(np.pi/4)
transformation = T @ R
transformed = (transformation @ origin.T).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0,0,0, 1,0,0, color='r', label='X')
ax.quiver(0,0,0, 0,1,0, color='g', label='Y')
ax.quiver(0,0,0, 0,0,1, color='b', label='Z')

ax.quiver(transformed[0,0], transformed[0,1], transformed[0,2], transformed[1,0]-transformed[0,0], transformed[1,1]-transformed[0,1], transformed[1,2]-transformed[0,2], color='r', linestyle='--')
ax.quiver(transformed[0,0], transformed[0,1], transformed[0,2], transformed[2,0]-transformed[0,0], transformed[2,1]-transformed[0,1], transformed[2,2]-transformed[0,2], color='g', linestyle='--')
ax.quiver(transformed[0,0], transformed[0,1], transformed[0,2], transformed[3,0]-transformed[0,0], transformed[3,1]-transformed[0,1], transformed[3,2]-transformed[0,2], color='b', linestyle='--')

ax.set_xlim([-1, 4]); ax.set_ylim([-1, 4]); ax.set_zlim([-1, 4])
plt.legend()
plt.title("Coordinate Frames and Transformation")
plt.savefig("output.png")
