# camera models
![Camera Models](output.png)

this part covers how a 3d point in the world projects onto a 2d sensor. this is exactly how our eyes or a phone camera works in real life.

## the math
we use the pinhole camera model. it uses an intrinsic matrix which has focal length (fx, fy) and optical center (cx, cy). we multiply the 3d point with this matrix to get 2d pixel coordinates.
the extrinsic matrix handles where the camera is in the world, while the intrinsic matrix handles how the camera projects the light.

## the script
the python code implements the intrinsic matrix n projects a simple 3d cube onto a 2d plane. everything is done with numpy n plotted with matplotlib so its easy n nderstandable. u can see the perspective effect where objects further away look smaller!
