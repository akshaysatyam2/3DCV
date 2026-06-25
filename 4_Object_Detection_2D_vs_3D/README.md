# 2d vs 3d object detection
![Comparison](output_comparison.png)

in traditional computer vision, we detect objects in 2d using flat bounding boxes. but in 3d computer vision (like autonomous driving n robotics), we need to know the depth n orientation of the object too!

## 2d detection (opencv)
the script first downloads an image from the web n uses a standard opencv haar cascade to detect a face. it draws a simple 2d rectangle `(x, y, w, h)` around it.

## 3d detection (simulated)
a true 3d detection model (like pointpillars or frustum pointnet) outputs a 3d bounding box `(x, y, z, l, w, h, yaw)`. since we dont have real lidar data here, the script simulates this by projecting a 3d box onto the 2d image using geometric offsets.

## comparison
i made the script save them seprately so u can see:
- **2d box:** `output_2d.png`
- **3d box:** `output_3d.png`

the combined plot `output_comparison.png` makes it really nderstandable how 3d gives us much more spatial awareness than just a flat 2d box!
