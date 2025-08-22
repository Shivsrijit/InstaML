# core/yolo_trainer.py
import subprocess
from pathlib import Path

def train_yolo(data_yaml: str, model="yolov8n.pt", epochs=50, imgsz=640, project="yolo_runs"):
    """Train YOLOv8 using Ultralytics CLI."""
    cmd = [
        "yolo", "task=detect", "mode=train",
        f"data={data_yaml}",
        f"model={model}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"project={project}"
    ]
    subprocess.run(cmd, check=True)
