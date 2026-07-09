import argparse
import cv2
import matplotlib.pyplot as plt
import os
from yolo_utils import get_yolo_model, detect_objects, estimate_3d_boxes, create_separate_figures

def main(file_path, display=True, save_output=False, focal_length=700.0, conf_threshold=0.5, model_path="../yolo26n.onnx"):
    """
    Main function to handle 2D to 3D object detection on images.
    
    Args:
        file_path (str): Location of the image file.
        display (bool): Whether to display using Matplotlib (not OpenCV).
        save_output (bool): Whether to save the output plot as an image.
        focal_length (float): Focal length in pixels. Default 700.0 is an approximation for standard webcams.
        conf_threshold (float): Detection confidence threshold. Default is 0.50.
        model_path (str): Path to YOLO ONNX model.
    """
    print(f"Loading image: {file_path}")
    img = cv2.imread(file_path)
    if img is None:
        print(f"Error: Could not read image at {file_path}")
        return
        
    img_h, img_w = img.shape[:2]
    
    # Load model
    print(f"Loading YOLO model from: {model_path}")
    net = get_yolo_model(model_path)
    
    # Detect objects (2D)
    print("Running detection...")
    boxes_2d = detect_objects(img, net, conf_threshold=conf_threshold)
    print(f"Detected {len(boxes_2d)} objects.")
    
    # Estimate 3D boxes
    boxes_3d = estimate_3d_boxes(boxes_2d, focal_length, img_w, img_h)
    
    # Generate Plots
    fig_2d, fig_3d = create_separate_figures(img, boxes_3d)
    
    # Save output if requested
    if save_output:
        base_name = os.path.basename(file_path)
        out_2d = "output_2d_" + base_name
        out_3d = "output_3d_" + base_name
        
        fig_2d.savefig(out_2d, bbox_inches='tight')
        fig_3d.savefig(out_3d, bbox_inches='tight')
        print(f"Saved 2D detection plot to {out_2d}")
        print(f"Saved 3D estimation plot to {out_3d}")
        
    # Display using matplotlib
    if display:
        plt.show()
    else:
        plt.close(fig_2d)
        plt.close(fig_3d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image 2D to 3D Object Detection using YOLO ONNX")
    parser.add_argument("--file", type=str, required=True, help="Path to input image")
    parser.add_argument("--no-display", action="store_true", help="Disable display of the output plot")
    parser.add_argument("--save", action="store_true", help="Save the output plot")
    parser.add_argument("--focal_length", type=float, default=700.0, help="Camera focal length in pixels")
    parser.add_argument("--conf", type=float, default=0.50, help="Detection confidence threshold")
    parser.add_argument("--model", type=str, default="../yolo26n.onnx", help="Path to YOLO ONNX model")
    
    args = parser.parse_args()
    main(args.file, not args.no_display, args.save, args.focal_length, args.conf, args.model)
