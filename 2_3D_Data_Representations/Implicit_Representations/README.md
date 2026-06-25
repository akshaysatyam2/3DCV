# implicit representations
![Implicit Representation](output.png)

instead of storing explicit geometry like meshes or points, we use a continuous function that maps 3d coordinates to values like distance or density.

## details
nerfs (neural radiance fields) use this! a neural network acts as the implicit function, taking an $(x, y, z)$ coordinate and outputting color and density. this allows for infinite resolution since its a continuous function!

## the script
i wrote a script showing a 2d slice of a signed distance field (sdf) for a sphere. the black line is the surface (distance=0). inside is negative distance, outside is positive. this helps visualize how implicit functions encode shapes.
