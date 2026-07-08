import os
import sys
import subprocess

topics = [
    ("1_Mathematical_Foundation/1_1_Coordinate_Frames", 
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

    ("1_Mathematical_Foundation/1_2_Camera_Models",
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

    ("1_Mathematical_Foundation/1_3_Epipolar_Geometry",
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

    ("2_3D_Data_Representations/2_1_Point_Clouds",
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

    ("2_3D_Data_Representations/2_2_Voxels",
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

    ("2_3D_Data_Representations/2_3_Meshes",
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

    ("2_3D_Data_Representations/2_4_Implicit_Representations",
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

    ("3_3D_Deep_Learning/3_1_PointNet",
"""import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Generate Synthetic 3D Shape Dataset (Sphere, Cube, Cylinder)
def generate_sphere(n_points=256):
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.arccos(np.random.uniform(-1, 1, n_points))
    r = 1.0 + np.random.normal(0, 0.05, n_points)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)

def generate_cube(n_points=256):
    points = []
    points_per_face = n_points // 6
    for i in range(6):
        face_pts = np.random.uniform(-1, 1, (points_per_face, 2))
        const = np.ones((points_per_face, 1)) * (1 if i % 2 == 0 else -1)
        if i < 2:
            pts = np.hstack([const, face_pts])
        elif i < 4:
            pts = np.hstack([face_pts[:, :1], const, face_pts[:, 1:]])
        else:
            pts = np.hstack([face_pts, const])
        points.append(pts)
    points = np.vstack(points)
    if len(points) < n_points:
        extra = np.random.uniform(-1, 1, (n_points - len(points), 3))
        points = np.vstack([points, extra])
    points += np.random.normal(0, 0.05, points.shape)
    return points

def generate_cylinder(n_points=256):
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(-1, 1, n_points)
    r = 1.0 + np.random.normal(0, 0.05, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)

def create_dataset(num_samples=100, n_points=256):
    data = []
    labels = []
    for _ in range(num_samples):
        data.append(generate_sphere(n_points))
        labels.append(0)
        data.append(generate_cube(n_points))
        labels.append(1)
        data.append(generate_cylinder(n_points))
        labels.append(2)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)

X_train, y_train = create_dataset(num_samples=60, n_points=256)
X_test, y_test = create_dataset(num_samples=20, n_points=256)

X_train_t = torch.tensor(X_train).transpose(1, 2)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test).transpose(1, 2)
y_test_t = torch.tensor(y_test)

# 2. PointNet Architecture
class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(PointNetClassifier, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        global_feat, argmax_idx = torch.max(x, dim=2)
        x = torch.relu(self.bn4(self.fc1(global_feat)))
        x = self.fc2(x)
        return x, argmax_idx

model = PointNetClassifier(num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 15
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

model.eval()
with torch.no_grad():
    test_outputs, test_argmax = model(X_test_t)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test_t).float().mean().item()
    print(f"Model trained! Accuracy: {accuracy*100:.1f}%")

cube_idx = np.where(y_test == 1)[0][0]
cube_points = X_test[cube_idx]
single_sample = X_test_t[cube_idx:cube_idx+1]

with torch.no_grad():
    pred, argmax_idx = model(single_sample)
    pred_class = torch.argmax(pred, dim=1).item()
    critical_indices = torch.unique(argmax_idx[0]).numpy()

class_names = ["Sphere", "Cube", "Cylinder"]

fig = plt.figure(figsize=(12, 5))
ax_loss = fig.add_subplot(121)
ax_loss.plot(range(1, epochs+1), losses, 'r-o', linewidth=2)
ax_loss.set_title("PointNet Training Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)

ax_3d = fig.add_subplot(122, projection='3d')
ax_3d.scatter(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2], c='gray', alpha=0.3, s=20, label='All Points')
critical_pts = cube_points[critical_indices]
ax_3d.scatter(critical_pts[:, 0], critical_pts[:, 1], critical_pts[:, 2], c='red', alpha=1.0, s=60, edgecolors='k', label='Critical Points')
title_str = f"Critical Points for {class_names[y_test[cube_idx]]}\\n(Predicted: {class_names[pred_class]})"
ax_3d.set_title(title_str)
ax_3d.legend()
plt.tight_layout()
plt.savefig("output.png", dpi=150)
print("Saved PointNet output visualization!")
""",
"""# pointnet
![PointNet](output.png)

pointnet is literally the "hello world" of 3d deep learning. standard cnns fail on point clouds cos they r unordered. pointnet fixes this!

## details
to process an unordered set of points, pointnet processes each point independently with a shared multi-layer perceptron (mlp), mapping them to a high-dimensional space. then it uses a symmetric mathematical function - specifically max pooling - to aggregate features across all points into a single global feature vector that represents the whole shape.

## the script
we generate a synthetic dataset of 3d shapes (spheres, cubes, n cylinders), train a mini pointnet model to classify them, n plot the training loss alongside the "critical points" that the model picked out to identify a cube. u can see the red points are the most important corners/edges that define the shape!
"""),

    ("4_Real_Time_Tooling/4_1_Open3D_Processing",
"""import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def has_avx_support():
    try:
        import platform
        if platform.system() != "Linux":
            return True
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line.split()
                    if "avx" in flags or "avx2" in flags:
                        return True
        return False
    except:
        return True

avx_supported = has_avx_support()

if avx_supported:
    try:
        import open3d as o3d
        USE_O3D = True
    except Exception as e:
        print(f"Failed to import Open3D: {e}. Falling back to NumPy.")
        USE_O3D = False
else:
    print("AVX instructions not supported by CPU. Running pipeline in NumPy/Sklearn fallback mode.")
    USE_O3D = False

np.random.seed(42)

# Ground plane points
n_ground = 1000
ground_x = np.random.uniform(-3, 3, n_ground)
ground_y = np.random.uniform(-3, 3, n_ground)
ground_z = np.random.normal(0, 0.02, n_ground)
ground = np.stack([ground_x, ground_y, ground_z], axis=1)

# Sphere obstacle
n_sphere = 300
theta = np.random.uniform(0, 2*np.pi, n_sphere)
phi = np.arccos(np.random.uniform(-1, 1, n_sphere))
r = 0.4 + np.random.normal(0, 0.02, n_sphere)
sphere_x = 1.0 + r * np.sin(phi) * np.cos(theta)
sphere_y = 1.0 + r * np.sin(phi) * np.sin(theta)
sphere_z = 0.4 + r * np.cos(phi)
sphere = np.stack([sphere_x, sphere_y, sphere_z], axis=1)

# Box obstacle
n_box = 300
box_x = np.random.uniform(-1.5, -0.7, n_box)
box_y = np.random.uniform(0.5, 1.5, n_box)
box_z = np.random.uniform(0, 0.8, n_box)
box = np.stack([box_x, box_y, box_z], axis=1)

points = np.vstack([ground, sphere, box])
points += np.random.normal(0, 0.01, points.shape)

if USE_O3D:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_size = 0.08
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = np.asarray(pcd_down.points)
    print(f"[Open3D] Downsampled points from {len(points)} to {len(downsampled_points)}")
    plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.06, ransac_n=3, num_iterations=100)
    ground_pcd = pcd_down.select_by_index(inliers)
    objects_pcd = pcd_down.select_by_index(inliers, invert=True)
    ground_pts = np.asarray(ground_pcd.points)
    object_pts = np.asarray(objects_pcd.points)
    labels = np.array(objects_pcd.cluster_dbscan(eps=0.3, min_points=8, print_progress=False))
else:
    voxel_size = 0.08
    voxel_coords = np.round(points / voxel_size).astype(int)
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    downsampled_points = points[unique_indices]
    print(f"[Fallback] Downsampled points from {len(points)} to {len(downsampled_points)}")
    n_pts = downsampled_points.shape[0]
    best_inliers = []
    for _ in range(100):
        idx = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = downsampled_points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)
        distances = np.abs(np.dot(downsampled_points, normal) + d)
        inliers = np.where(distances < 0.06)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    inliers_set = set(best_inliers)
    outliers = [i for i in range(n_pts) if i not in inliers_set]
    ground_pts = downsampled_points[best_inliers]
    object_pts = downsampled_points[outliers]
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=0.3, min_samples=8).fit(object_pts)
    labels = db.labels_

max_label = labels.max()
print(f"Detected {max_label + 1} distinct objects on the road!")

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap='viridis', s=2, alpha=0.5)
ax1.set_title("Raw Lidar Input Point Cloud")
ax1.set_xlim([-3, 3]); ax1.set_ylim([-3, 3]); ax1.set_zlim([-0.2, 1.5])

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(ground_pts[:,0], ground_pts[:,1], ground_pts[:,2], c='lightgreen', s=5, alpha=0.3, label='Ground (RANSAC)')

colors = plt.get_cmap("tab10")
for i in range(max_label + 1):
    cluster_idx = np.where(labels == i)[0]
    cluster_pts = object_pts[cluster_idx]
    ax2.scatter(cluster_pts[:,0], cluster_pts[:,1], cluster_pts[:,2], color=colors(i % 10), s=15, alpha=0.9, label=f'Obstacle {i+1}')

noise_idx = np.where(labels == -1)[0]
if len(noise_idx) > 0:
    noise_pts = object_pts[noise_idx]
    ax2.scatter(noise_pts[:,0], noise_pts[:,1], noise_pts[:,2], c='gray', s=5, alpha=0.2, label='Noise')

ax2.set_title("Processed: Road Segmented n Objects Clustered")
ax2.set_xlim([-3, 3]); ax2.set_ylim([-3, 3]); ax2.set_zlim([-0.2, 1.5])
ax2.legend()
plt.tight_layout()
plt.savefig("output.png", dpi=150)
print("Saved Open3D processing visualization!")
""",
"""# real-time tooling n hardware processing
![Open3D Output](output.png)

since 3d point clouds can have millions of points, real-time performance needs more than just python. we use open3d (which has an optimized c++ core) to do high-speed geometry operations.

## our real-time lidar pipeline
1. **voxel downsampling:** average points in tiny 3d grids (voxels) to drastically cut down point count while keeping the shapes intact.
2. **ransac plane segmentation:** mathematically fit a flat plane to the road. once we identify the road, we filter it out to isolate actual obstacles.
3. **dbscan clustering:** group the remaining points based on density. if points r close together, they r grouped into the same object (like a car or a pedestrian).

## the script
we simulate a lidar scan w/ a road n two obstacles. the script runs the entire pipeline, prints the details, n saves a 3d plot showing the segmented road (green) n the clustered obstacles (colored)!
""")
]

for folder, script_code, readme_code in topics:
    script_path = os.path.join(folder, "script.py")
    readme_path = os.path.join(folder, "README.md")
    
    os.makedirs(folder, exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write(script_code)
        
    with open(readme_path, "w") as f:
        f.write(readme_code)
        
    subprocess.run([sys.executable, "script.py"], cwd=os.path.join(os.getcwd(), folder))

    old_doc = os.path.join(folder, "doc.md")
    if os.path.exists(old_doc):
        os.remove(old_doc)

print("done generating all files and images.")
