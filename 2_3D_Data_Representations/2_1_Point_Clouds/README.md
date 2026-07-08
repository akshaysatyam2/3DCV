# point clouds
![Point Cloud](output.png)

point clouds r just unordered sets of (x,y,z) coordinates. they r usually the raw output from lidar or depth cameras in real world applications like autonomous driving.

## details
a point cloud is simply an $N \times 3$ matrix. since its unordered, changing the row order doesnt change the 3d shape at all. they are memory efficient for sparse structures but lack connectivity info.

## the script
the script generates a random point cloud in the shape of a sphere using spherical coordinates n plots it in 3d. the color maps to the z-axis. simple n nderstandable!
