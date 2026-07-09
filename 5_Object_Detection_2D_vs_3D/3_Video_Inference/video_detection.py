import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from yolo_utils import get_yolo_model, detect_objects, estimate_3d_boxes, create_3d_plot_figure

def get_img_from_fig(fig, dpi=100):
    """Converts a matplotlib figure to an OpenCV BGR image."""
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    # Roll the ARGB buffer to RGBA
    buf = np.roll(buf, 3, axis=2)
    # Convert RGBA to BGR for OpenCV
    img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return img

def main(file_path, display=True, save_output=False, focal_length=700.0, conf_threshold=0.5, model_path="../yolo26n.onnx"):
    """
    Main function to handle 2D to 3D object detection on videos/RTSP streams.
    
    Args:
        file_path (str): Location of the video file or RTSP stream link.
        display (bool): Whether to display using Matplotlib (not OpenCV).
        save_output (bool): Whether to save the output video.
        focal_length (float): Focal length in pixels.
        conf_threshold (float): Detection confidence threshold. Default is 0.50.
        model_path (str): Path to YOLO ONNX model.
    """
    print(f"Loading YOLO model from: {model_path}")
    net = get_yolo_model(model_path)
    
    print(f"Opening video source: {file_path}")
    # Handle RTSP or webcam if input is a digit (like "0")
    if file_path.isdigit():
        file_path = int(file_path)
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {file_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle RTSP sometimes returning NaN or 0
        fps = 25.0
        
    video_writer = None
    if save_output:
        out_filename = "output_video.mp4"
        # We need to set the video writer resolution based on the matplotlib figure size
        # A 14x7 inch figure at 100 DPI is 1400x700 pixels
        out_w, out_h = 1400, 700 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (out_w, out_h))
        print(f"Saving output video to {out_filename}")

    if display:
        plt.ion() # Interactive mode on for live updating
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or cannot read the frame.")
                break
                
            frame_count += 1
            
            # Detect objects
            boxes_2d = detect_objects(frame, net, conf_threshold=conf_threshold)
            
            # Estimate 3D boxes
            boxes_3d = estimate_3d_boxes(boxes_2d, focal_length, width, height)
            
            # Generate Plot
            fig = create_3d_plot_figure(frame, boxes_3d)
            
            # If we need to save the output, extract image from matplotlib figure
            if save_output and video_writer:
                fig.set_size_inches(14, 7)
                fig.set_dpi(100)
                frame_out = get_img_from_fig(fig, dpi=100)
                video_writer.write(frame_out)
                
            if display:
                plt.draw()
                plt.pause(0.001)
                
            # Close the figure so we don't leak memory and create thousands of figures
            plt.close(fig)
            
    except KeyboardInterrupt:
        print("Interrupted by user.")
        
    finally:
        cap.release()
        if save_output and video_writer:
            video_writer.release()
        if display:
            plt.ioff()
        print(f"Processed {frame_count} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video 2D to 3D Object Detection using YOLO ONNX")
    parser.add_argument("--file", type=str, required=True, help="Path to input video or RTSP stream link")
    parser.add_argument("--no-display", action="store_true", help="Disable display of the output plot (runs faster)")
    parser.add_argument("--save", action="store_true", help="Save the output video")
    parser.add_argument("--focal_length", type=float, default=700.0, help="Camera focal length in pixels")
    parser.add_argument("--conf", type=float, default=0.50, help="Detection confidence threshold")
    parser.add_argument("--model", type=str, default="../yolo26n.onnx", help="Path to YOLO ONNX model")
    
    args = parser.parse_args()
    main(args.file, not args.no_display, args.save, args.focal_length, args.conf, args.model)
