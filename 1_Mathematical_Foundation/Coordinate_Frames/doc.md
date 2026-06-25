# Coordinate Frames and Transformations

in 3d computer vision, we need to represent objects in 3d space. we use coordinate frames to define where things r.

## mathematical representation
we use 4x4 matrices in $SE(3)$ to represent rotations and translations together. this is called homogeneous coordinates.
a point is represented as $(x, y, z, 1)$.

## what the script does
the python script creates simple rotation n translation matrices using numpy. it shows how to apply these transformations to a 3d point using matrix multiplication so it is completely nderstandable.
