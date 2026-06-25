import numpy as np

def create_intrinsic_matrix(fx, fy, cx, cy):
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

def project_point(K, point_3d):
    # point_3d is [x, y, z]
    point_2d_homogeneous = K @ point_3d
    # normalize
    u = point_2d_homogeneous[0] / point_2d_homogeneous[2]
    v = point_2d_homogeneous[1] / point_2d_homogeneous[2]
    return u, v

K = create_intrinsic_matrix(800, 800, 320, 240)
pt_3d = np.array([10, 5, 20])
u, v = project_point(K, pt_3d)
print(f"3d point {pt_3d} projects to pixel ({u}, {v})")
