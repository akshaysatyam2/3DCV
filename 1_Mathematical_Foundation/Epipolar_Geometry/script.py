import numpy as np

def compute_fundamental_matrix(pts1, pts2):
    # this is a simple 8-point algorithm implementation
    A = []
    for p1, p2 in zip(pts1, pts2):
        u1, v1 = p1[0], p1[1]
        u2, v2 = p2[0], p2[1]
        A.append([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1])
    A = np.array(A)
    
    # SVD
    U, S, Vh = np.linalg.svd(A)
    F = Vh[-1].reshape(3, 3)
    
    # enforce rank 2
    U_f, S_f, Vh_f = np.linalg.svd(F)
    S_f[2] = 0
    F = U_f @ np.diag(S_f) @ Vh_f
    return F

# dummy points
pts1 = np.random.rand(8, 2) * 100
pts2 = np.random.rand(8, 2) * 100

F = compute_fundamental_matrix(pts1, pts2)
print("computed fundamental matrix:")
print(F)
