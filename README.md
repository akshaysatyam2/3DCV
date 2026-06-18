# 3DCV

This repository tracks my journey and projects as I learn and build in the field of 3D Computer Vision (3DCV).

## Learning Approach

### 1. Master the Mathematical Foundation (Multi-View Geometry)
Before touching a neural network, it is essential to understand how the 3D world projects onto a 2D sensor. This is entirely based on linear algebra and matrix operations.
- **Coordinate Frames and Transformations:** Represent 3D rotations and translations using matrices in $SE(3)$ and quaternions.
- **Camera Models:** Understand the pinhole camera model. Manually construct the intrinsic matrix (focal length and optical center) and extrinsic matrix (the camera's rotation $R$ and translation $t$ in world space).
- **Epipolar Geometry:** The core of stereovision and 3D reconstruction. Understand how to map a point in one camera view to a line in another using the Fundamental and Essential matrices.
- **Project Idea:** Implement camera calibration or a fundamental matrix calculator entirely from scratch using NumPy. No OpenCV for this step—code the Singular Value Decomposition (SVD) and matrix multiplications manually.

### 2. Understand 3D Data Representations
Unlike 2D images which are always grid-like matrices, 3D data requires choosing a representation before building a pipeline. Each has different mathematical properties:
- **Point Clouds:** Unordered sets of $(x, y, z)$ coordinates. Often raw sensor data from LiDAR or depth cameras. Represented as an $N \times 3$ matrix.
- **Voxels:** 3D pixel grids. Conceptually easy because 3D convolutions can be applied to them, but incredibly memory-intensive (most of a 3D space is empty).
- **Meshes:** Vertices and faces defining a surface. Great for computer graphics, but mathematically complex for neural networks to process.
- **Implicit Representations:** Functions (like Neural Radiance Fields) that map a continuous 3D coordinate directly to a density or color value using a multi-layer perceptron.

### 3. The "Hello World" of 3D Deep Learning
Once the math and data structures are understood, the next step is building the first 3D network. The perfect starting point is PointNet.
- Standard Convolutional Neural Networks (CNNs) fail on point clouds because point clouds are orderless (shuffling the rows of an $N \times 3$ point cloud matrix doesn't change the 3D shape). 
- PointNet solves this by using a symmetric mathematical function—specifically, Max Pooling—to aggregate features across all points regardless of their order.
- **Project Idea:** Build PointNet from scratch in PyTorch. It relies almost entirely on shared Multi-Layer Perceptrons (MLPs) and matrix multiplications. Train it to classify 3D CAD models from the ModelNet40 dataset.

### 4. Real-Time Tooling and Hardware Processing
Because 3D point clouds can contain millions of points, real-time performance requires moving beyond Python.
- **Open3D:** A modern library that bridges the gap beautifully. It has excellent Python bindings for prototyping, but its core is written in highly optimized C++, making it great for production pipelines.
