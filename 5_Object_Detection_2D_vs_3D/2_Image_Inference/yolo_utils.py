import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def get_yolo_model(model_path="../yolo26n.onnx"):
    """Loads the YOLOv8 ONNX model using OpenCV DNN with hardware acceleration."""
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found.")
        print("Please ensure you have exported it via: from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')")
    
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # Maximize performance: check for CUDA, then OpenVINO, then fallback to CPU
    try:
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("CUDA detected! Enabling NVIDIA GPU acceleration.")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            return net
    except Exception:
        pass
        
    try:
        print("No CUDA found. Attempting OpenVINO (Inference Engine)...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        print("OpenVINO unavailable. Falling back to standard OpenCV CPU backend.")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    return net

def process_yolo_output(outputs, img_w, img_h, conf_threshold=0.5):
    """Processes YOLO ONNX output and returns bounding boxes, class ids, and scores."""
    output = outputs[0]
    
    boxes = []
    scores = []
    class_ids = []
    
    if len(output.shape) == 2 and output.shape[1] == 6:
        for row in output:
            x1_640, y1_640, x2_640, y2_640, score, class_id = row
            if score > conf_threshold:
                x1 = int(x1_640 / 640 * img_w)
                y1 = int(y1_640 / 640 * img_h)
                x2 = int(x2_640 / 640 * img_w)
                y2 = int(y2_640 / 640 * img_h)
                w = x2 - x1
                h = y2 - y1
                boxes.append([x1, y1, w, h])
                scores.append(float(score))
                class_ids.append(int(class_id))
        result_boxes = []
        for i in range(len(boxes)):
            result_boxes.append((boxes[i], class_ids[i], scores[i]))
        return result_boxes

    if output.shape[0] < output.shape[1]:
        output = output.T 
    
    for row in output:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]
        
        if score > conf_threshold:
            cx, cy, w, h = row[:4]
            x1 = int((cx - w / 2) / 640 * img_w)
            y1 = int((cy - h / 2) / 640 * img_h)
            width = int(w / 640 * img_w)
            height = int(h / 640 * img_h)
            
            boxes.append([x1, y1, width, height])
            scores.append(float(score))
            class_ids.append(class_id)
            
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.4)
    result_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            result_boxes.append((boxes[i], class_ids[i], scores[i]))
    return result_boxes

def detect_objects(img, net, conf_threshold=0.5):
    """Runs a forward pass of YOLOv8 and processes the output."""
    img_h, img_w = img.shape[:2]
    # YOLOv8 standard input is 640x640, normalized by 1/255.0, RGB
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    return process_yolo_output(outputs, img_w, img_h, conf_threshold)

def estimate_3d_boxes(boxes, focal_length, img_w, img_h, default_obj_width=1.0):
    """Estimates 3D box parameters from 2D boxes using camera focal length."""
    boxes_3d = []
    cx_img, cy_img = img_w / 2, img_h / 2
    for box, cls, score in boxes:
        x, y, w, h = box
        # Avoid division by zero
        if w == 0 or h == 0:
            continue
            
        # Z = f * W / w_pixels
        Z = focal_length * default_obj_width / w
        
        # Center of bbox in pixels
        cx = x + w / 2
        cy = y + h / 2
        
        # 3D position (X, Y) at estimated Depth Z
        X = (cx - cx_img) * Z / focal_length
        Y = (cy - cy_img) * Z / focal_length
        
        # Height is proportional to pixel height vs width
        H = default_obj_width * (h / w)
        D = default_obj_width # Assume depth is roughly equal to width
        
        # Calculate 3D corners relative to camera
        half_w, half_h, half_d = default_obj_width/2, H/2, D/2
        corners_3d = [
            (X - half_w, Y - half_h, Z - half_d), (X + half_w, Y - half_h, Z - half_d),
            (X + half_w, Y + half_h, Z - half_d), (X - half_w, Y + half_h, Z - half_d),
            (X - half_w, Y - half_h, Z + half_d), (X + half_w, Y - half_h, Z + half_d),
            (X + half_w, Y + half_h, Z + half_d), (X - half_w, Y + half_h, Z + half_d)
        ]
        
        # Project back to 2D image plane
        corners_2d = []
        for cX, cY, cZ in corners_3d:
            u = cX * focal_length / cZ + cx_img
            v = cY * focal_length / cZ + cy_img
            corners_2d.append((u, v))
        
        boxes_3d.append({
            'center': (X, Y, Z),
            'size': (default_obj_width, H, D),
            'class': cls,
            'score': score,
            'box_2d': box,
            'corners_2d': corners_2d
        })
    return boxes_3d

def create_separate_figures(img, boxes_3d):
    """Creates two separate Matplotlib figures for 2D and 3D views."""
    # --- 2D Figure ---
    fig_2d = plt.figure(figsize=(8, 8))
    ax1 = fig_2d.add_subplot(111)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for i, b in enumerate(boxes_3d):
        x, y, w, h = b['box_2d']
        
        # Generate a distinct color for each object instance
        np.random.seed(i * 12345 + b['class']) 
        obj_color = tuple(np.random.rand(3))
        
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=obj_color, facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x, max(y-5, 0), f"Cls: {b['class']} | {b['score']:.2f}", color='white', fontsize=10, backgroundcolor=obj_color)
        
        # Draw projected 3D wireframe on the 2D image
        if 'corners_2d' in b:
            c = b['corners_2d']
            edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            for pt1_idx, pt2_idx in edges:
                ax1.plot([c[pt1_idx][0], c[pt2_idx][0]], [c[pt1_idx][1], c[pt2_idx][1]], color=obj_color, linewidth=1.5, alpha=0.9)
    
    ax1.imshow(img_rgb)
    ax1.set_title("2D YOLO Detections")
    ax1.axis("off")
    
    # --- 3D Figure ---
    fig_3d = plt.figure(figsize=(8, 8))
    ax2 = fig_3d.add_subplot(111, projection='3d')
    ax2.set_title("3D Bounding Boxes (Estimated)")
    
    # Camera at origin
    ax2.scatter(0, 0, 0, color='k', marker='^', s=150, label='Camera')
    
    max_z = 10 # Default max limit for depth
    
    for b in boxes_3d:
        X, Y, Z = b['center']
        W, H, D = b['size']
        if Z > max_z:
            max_z = Z + 5
            
        half_w, half_h, half_d = W/2, H/2, D/2
        corners = np.array([
            [X - half_w, Y - half_h, Z - half_d],
            [X + half_w, Y - half_h, Z - half_d],
            [X + half_w, Y + half_h, Z - half_d],
            [X - half_w, Y + half_h, Z - half_d],
            [X - half_w, Y - half_h, Z + half_d],
            [X + half_w, Y - half_h, Z + half_d],
            [X + half_w, Y + half_h, Z + half_d],
            [X - half_w, Y + half_h, Z + half_d],
        ])
        
        faces = [
            [corners[0], corners[1], corners[2], corners[3]], # Front
            [corners[4], corners[5], corners[6], corners[7]], # Back
            [corners[0], corners[1], corners[5], corners[4]], # Bottom
            [corners[2], corners[3], corners[7], corners[6]], # Top
            [corners[1], corners[2], corners[6], corners[5]], # Right
            [corners[0], corners[3], corners[7], corners[4]]  # Left
        ]
        
        # Recreate the exact same color using the same seed for the 3D plot
        np.random.seed(i * 12345 + b['class'])
        obj_color = tuple(np.random.rand(3))
        
        poly3d = Poly3DCollection(faces, alpha=0.4, facecolors=obj_color, edgecolors=obj_color, linewidths=1.5)
        ax2.add_collection3d(poly3d)
        
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_zlabel('Z (Depth, meters)')
    
    # Set standard limits
    ax2.set_xlim([-max_z/2, max_z/2])
    ax2.set_ylim([-max_z/2, max_z/2])
    ax2.set_zlim([0, max_z])
    
    ax2.legend()
    fig_2d.tight_layout()
    fig_3d.tight_layout()
    return fig_2d, fig_3d
