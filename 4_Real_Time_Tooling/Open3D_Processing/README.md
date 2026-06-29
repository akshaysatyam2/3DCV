# real-time tooling n hardware processing
![Open3D Output](output.png)

since 3d point clouds can have millions of points, real-time performance needs more than just python. we use open3d (which has an optimized c++ core) to do high-speed geometry operations.

## our real-time lidar pipeline
1. **voxel downsampling:** average points in tiny 3d grids (voxels) to drastically cut down point count while keeping the shapes intact.
2. **ransac plane segmentation:** mathematically fit a flat plane to the road. once we identify the road, we filter it out to isolate actual obstacles.
3. **dbscan clustering:** group the remaining points based on density. if points r close together, they r grouped into the same object (like a car or a pedestrian).

## the script
we simulate a lidar scan w/ a road n two obstacles. the script runs the entire pipeline, prints the details, n saves a 3d plot showing the segmented road (green) n the clustered obstacles (colored)!
