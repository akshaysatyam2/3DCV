# camera models

this part covers how a 3d point in the world projects onto a 2d sensor. this is like how our eyes or a phone camera works.

## the math
we use the pinhole camera model. it uses an intrinsic matrix which has focal length (fx, fy) and optical center (cx, cy). we multiply the 3d point with this matrix to get 2d pixel coordinates.

## the script
the python code implements the intrinsic matrix n projects a simple 3d point onto a 2d plane. everything is done with numpy so its easy n nderstandable.
