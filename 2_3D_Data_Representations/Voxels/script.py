import numpy as np

def create_voxel_grid(size):
    """creates a 3d voxel grid of given size."""
    return np.zeros((size, size, size), dtype=int)

def set_voxel(grid, x, y, z):
    if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2]:
        grid[x, y, z] = 1

voxel_grid = create_voxel_grid(10)
set_voxel(voxel_grid, 5, 5, 5)
print(f"voxel grid shape: {voxel_grid.shape}")
print(f"is voxel at (5,5,5) active? {voxel_grid[5,5,5] == 1}")
