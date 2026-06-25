# pointnet
![PointNet](output.png)

pointnet is literally the "hello world" of 3d deep learning. standard cnns fail on point clouds cos they r unordered. pointnet fixes this!

## details
to process an unordered set of points, pointnet processes each point independently with a shared multi-layer perceptron (mlp), mapping them to a high-dimensional space. then it uses a symmetric mathematical function—specifically max pooling—to aggregate features across all points into a single global feature vector that represents the whole shape.

## the script
the script simulates the max pooling step. it shows how out of many input points (gray), only a few specific structural points (red/blue) actually contribute to the final global feature vector after max pooling! those r the critical points that define the shape.
