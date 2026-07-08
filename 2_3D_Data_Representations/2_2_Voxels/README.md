# voxels
![Voxels](output.png)

voxels r like 3d pixels. they form a dense grid where each cell is either empty or filled (or has a density value).

## details
its conceptually easy cos u can apply 3d convolutions on them directly just like 2d images! but they take up a ton of memory cos most of a 3d space is empty (its $O(N^3)$ complexity).

## the script
i made a 3d numpy array to act as a voxel grid and filled it to create a hollow box structure. the script then renders the voxels using matplotlibs built in voxel plotting. really shows how blocks build up a shape!
