import numpy as np

def simplified_pointnet(point_cloud):
    # point_cloud is Nx3
    # apply a shared MLP (here just a random weight matrix for simplicity)
    W = np.random.rand(3, 64)
    features = point_cloud @ W # now Nx64
    
    # apply symmetric function (max pooling) over all points
    global_feature = np.max(features, axis=0) # shape: (64,)
    
    return global_feature

pc = np.random.rand(100, 3)
global_feat = simplified_pointnet(pc)
print(f"input point cloud: {pc.shape}")
print(f"global feature shape after max pooling: {global_feat.shape}")
