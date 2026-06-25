import os
import subprocess

topics = [
    ("1_Mathematical_Foundation/Coordinate_Frames", 
"""import numpy as np
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
""",
"""# coordinate frames n transformations
![Coordinate Frames](output.png)

in 3d computer vision, we need to represent objects in 3d space accurately. we use coordinate frames to define where things r.

## mathematical representation
we use 4x4 matrices in $SE(3)$ to represent rotations and translations together. this is called homogeneous coordinates.
a point is represented as $(x, y, z, 1)$. this allows us to combine rotation and translation into a single matrix multiplication!

## what the script does
the python script creates simple rotation n translation matrices using numpy. it then applies these transformations to a 3d coordinate frame and plots the original (solid lines) n the transformed frame (dashed lines) using matplotlib.
this is super useful cos everything in 3d space relates to these transformations.
"""),

    ("1_Mathematical_Foundation/Camera_Models",
"""import numpy as np
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
""",
"""# camera models
![Camera Models](output.png)

this part covers how a 3d point in the world projects onto a 2d sensor. this is exactly how our eyes or a phone camera works in real life.

## the math
we use the pinhole camera model. it uses an intrinsic matrix which has focal length (fx, fy) and optical center (cx, cy). we multiply the 3d point with this matrix to get 2d pixel coordinates.
the extrinsic matrix handles where the camera is in the world, while the intrinsic matrix handles how the camera projects the light.

## the script
the python code implements the intrinsic matrix n projects a simple 3d cube onto a 2d plane. everything is done with numpy n plotted with matplotlib so its easy n nderstandable. u can see the perspective effect where objects further away look smaller!
"""),

    ("1_Mathematical_Foundation/Epipolar_Geometry",
"""import numpy as np
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
""",
"""# epipolar geometry
![Epipolar Geometry](output.png)

this is the core of stereovision n 3d reconstruction. it tells us how to map a point in one camera view to a line in another view.

## the math
it uses fundamental and essential matrices. the fundamental matrix relates corresponding points in stereo images. we can calculate it using the 8-point algorithm with singular value decomposition (svd).
epipolar lines r the projection of the camera ray from one view into the other.

## the script
i wrote a script that simulates a stereo camera setup! it creates random 3d points and projects them into two different camera views (left n right). u can see how the points shift horizontally, just like how our two eyes perceive depth.
"""),

    ("2_3D_Data_Representations/Point_Clouds",
"""import numpy as np
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
""",
"""# point clouds
![Point Cloud](output.png)

point clouds r just unordered sets of (x,y,z) coordinates. they r usually the raw output from lidar or depth cameras in real world applications like autonomous driving.

## details
a point cloud is simply an $N \\times 3$ matrix. since its unordered, changing the row order doesnt change the 3d shape at all. they are memory efficient for sparse structures but lack connectivity info.

## the script
the script generates a random point cloud in the shape of a sphere using spherical coordinates n plots it in 3d. the color maps to the z-axis. simple n nderstandable!
"""),

    ("2_3D_Data_Representations/Voxels",
"""import numpy as np
import matplotlib.pyplot as plt

grid = np.zeros((10, 10, 10), dtype=bool)
grid[2:8, 2:8, 2:8] = True
grid[3:7, 3:7, 3:7] = False

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(grid, edgecolor='k')
ax.set_title("Voxel Grid Representation")
plt.savefig("output.png")
""",
"""# voxels
![Voxels](output.png)

voxels r like 3d pixels. they form a dense grid where each cell is either empty or filled (or has a density value).

## details
its conceptually easy cos u can apply 3d convolutions on them directly just like 2d images! but they take up a ton of memory cos most of a 3d space is empty (its $O(N^3)$ complexity).

## the script
i made a 3d numpy array to act as a voxel grid and filled it to create a hollow box structure. the script then renders the voxels using matplotlibs built in voxel plotting. really shows how blocks build up a shape!
"""),

    ("2_3D_Data_Representations/Meshes",
"""import numpy as np
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
""",
"""# meshes
![Meshes](output.png)

meshes r made of vertices n faces defining a 3d surface. they r the standard for computer graphics n gaming.

## details
a mesh stores a list of 3d vertices n a list of faces (usually triangles) that connect them. this explicitly defines the surface topology. really great for rendering, but harder for neural nets to process directly cos of irregular graph structures.

## the script
the script creates a simple pyramid mesh from scratch using vertices n face definitions, n plots it as a 3d polygon collection. 
"""),

    ("2_3D_Data_Representations/Implicit_Representations",
"""import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2) - 1.0

plt.figure()
cp = plt.contourf(X, Y, Z, levels=20, cmap='RdBu')
plt.colorbar(cp, label='Distance to surface')
plt.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
plt.title("Implicit Representation (Signed Distance Field)")
plt.savefig("output.png")
""",
"""# implicit representations
![Implicit Representation](output.png)

instead of storing explicit geometry like meshes or points, we use a continuous function that maps 3d coordinates to values like distance or density.

## details
nerfs (neural radiance fields) use this! a neural network acts as the implicit function, taking an $(x, y, z)$ coordinate and outputting color and density. this allows for infinite resolution since its a continuous function!

## the script
i wrote a script showing a 2d slice of a signed distance field (sdf) for a sphere. the black line is the surface (distance=0). inside is negative distance, outside is positive. this helps visualize how implicit functions encode shapes.
"""),

    ("3_3D_Deep_Learning/PointNet",
"""import numpy as np
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
""",
"""# pointnet
![PointNet](output.png)

pointnet is literally the "hello world" of 3d deep learning. standard cnns fail on point clouds cos they r unordered. pointnet fixes this!

## details
to process an unordered set of points, pointnet processes each point independently with a shared multi-layer perceptron (mlp), mapping them to a high-dimensional space. then it uses a symmetric mathematical function - specifically max pooling - to aggregate features across all points into a single global feature vector that represents the whole shape.

## the script
the script simulates the max pooling step. it shows how out of many input points (gray), only a few specific structural points (red/blue) actually contribute to the final global feature vector after max pooling! those r the critical points that define the shape.
""")
]

for folder, script_code, readme_code in topics:
    script_path = os.path.join(folder, "script.py")
    readme_path = os.path.join(folder, "README.md")
    
    with open(script_path, "w") as f:
        f.write(script_code)
        
    with open(readme_path, "w") as f:
        f.write(readme_code)
        
    subprocess.run(["python", "script.py"], cwd=os.path.join(os.getcwd(), folder))

    old_doc = os.path.join(folder, "doc.md")
    if os.path.exists(old_doc):
        os.remove(old_doc)

print("done generating all files and images.")
