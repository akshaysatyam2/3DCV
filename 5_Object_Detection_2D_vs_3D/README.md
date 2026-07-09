# 2d vs 3d object detection

![Comparison](1_OpenCV_2D_vs_3D/output_comparison.png)

in traditional computer vision, we detect objects in 2d using flat bounding boxes. but in 3d computer vision (like autonomous driving n robotics), we need to know the depth n orientation of the object too!

recently updated this folder to b highly modular n production-ready. organized stuff into three main dirs for better ordering n accessibility:

## 1_OpenCV_2D_vs_3D (the basics)
this holds my original scripts that just do a basic 2d vs 3d comparison using opencv haar cascades.
- u get to see the standard 2d box vs pseudo-3d vs true `cv2.projectPoints` projection.
- check out `output_comparison.png` to see the 3-way difference.

## 2_Image_Inference (yolo onnx inference)
moved onto modern deep learning! this folder runs a yolo26 model exported to onnx format on static images.
- uses a custom `yolo_utils.py` that manually decodes the `(1, 300, 6)` output tensor from the onnx graph (cos nms is built-in!).
- projects the 2d detections into 3d space using camera focal length.
- renders the 3d bounding boxes in a dedicated matplotlib 3d plot space.

**Step 1: 2D Detection with YOLO26:**
![yolo26 2d detection](2_Image_Inference/output_2d_pet.jpg)

**Step 2: 3D Depth Estimation:**
![yolo26 3d estimation](2_Image_Inference/output_3d_pet.jpg)
**how to run:**
```bash
cd 2_Image_Inference
python image_detection.py --file pet.jpg --save
```

## 3_Video_Inference (rtsp n mp4 streams)
exact same modular logic as images, but built for real-time video feeds n rtsp streams.
- takes a video or stream link, processes it frame by frame, n updates the 3d plot in real-time.
- if u pass `--save`, it intercepts the matplotlib buffer n writes it directly to an mp4 file without popping up a gui window (super clean for headless servers).
**how to run:**
```bash
cd 3_Video_Inference
python video_detection.py --file 14968619_2160_3840_30fps.mp4 --no-display --save
```

## color coding details (for the old opencv script)
to make it super understandable what is actually detected vs what is simulated depth:
- **red square:** this is the original 2d detection box (the front face). it shows exactly where the object was found in the flat image.
- **blue lines n square:** this is the generated 3d depth n the back face of the box. it represents the simulated 3d volume projecting out from the 2d detection.
