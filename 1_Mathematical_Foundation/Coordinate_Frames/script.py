import numpy as np

def create_rotation_matrix_z(theta):
    """Creates a rotation matrix around Z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translate(dx, dy, dz):
    """Creates a translation matrix."""
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

# Example usage
point = np.array([1, 0, 0, 1])
T = translate(1, 2, 3)
R = create_rotation_matrix_z(np.pi/2)
transformation = T @ R
transformed_point = transformation @ point
print("Original Point:", point[:3])
print("Transformed Point:", transformed_point[:3])
