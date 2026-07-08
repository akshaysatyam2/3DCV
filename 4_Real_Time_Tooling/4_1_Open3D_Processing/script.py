import numpy as np
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
