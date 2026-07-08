import numpy as np
import matplotlib.pyplot as plt

def create_intrinsic_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

points_3d = []
for x in [-1, 1]:
    for y in [-1, 1]:
        for z in [4, 6]:
            points_3d.append([x, y, z])
points_3d = np.array(points_3d).T

K = create_intrinsic_matrix(800, 800, 320, 240)
points_2d_hom = K @ points_3d
points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]

plt.figure(figsize=(6,4))
plt.scatter(points_2d[0], points_2d[1], c='r')
for i in range(4):
    plt.plot([points_2d[0,i], points_2d[0,i+4]], [points_2d[1,i], points_2d[1,i+4]], 'b-')
plt.xlim(0, 640)
plt.ylim(0, 480)
plt.gca().invert_yaxis()
plt.title("Pinhole Camera Projection of a 3D Cube")
plt.savefig("output.png")
