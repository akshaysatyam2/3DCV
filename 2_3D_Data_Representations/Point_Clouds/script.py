import numpy as np

def generate_random_point_cloud(num_points):
    """generates a random Nx3 point cloud."""
    return np.random.rand(num_points, 3)

pc = generate_random_point_cloud(100)
print(f"generated point cloud shape: {pc.shape}")
print("first 3 points:")
print(pc[:3])
