# epipolar geometry

this is the core of stereovision n 3d reconstruction. it tells us how to map a point in one camera view to a line in another view.

## the math
it uses fundamental and essential matrices. the fundamental matrix relates corresponding points in stereo images. we can calculate it using the 8-point algorithm with singular value decomposition (svd).

## the script
i wrote a simple fundamental matrix calculator from scratch using numpy. it takes 8 point correspondences n uses svd to find the matrix. no opencv used here, all manual math!
