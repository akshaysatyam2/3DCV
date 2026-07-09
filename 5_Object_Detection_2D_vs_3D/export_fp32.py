from ultralytics import YOLO
import shutil
import os

print("Downloading/Loading yolov8n.pt...")
model = YOLO("yolov8n.pt")

print("Exporting to ONNX in explicitly FP32 precision...")
# half=False enforces FP32 precision
exported_path = model.export(format="onnx", half=False, imgsz=640)

target_path = "yolo26n.onnx"
if os.path.exists(target_path):
    os.remove(target_path)

shutil.move(exported_path, target_path)
print(f"Successfully exported FP32 model and saved as {target_path}!")
