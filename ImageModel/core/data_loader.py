import pandas as pd
import os
from pathlib import Path
from typing import Tuple

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def load_image_folder(folder_path: str) -> Tuple[list, list]:
    """Load images and labels from a folder structured as class_name/image.jpg."""
    from PIL import Image
    images, labels = [], []
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    for class_dir in folder.iterdir():
        if class_dir.is_dir():
            for img_file in class_dir.glob("*.*"):
                try:
                    img = Image.open(img_file).convert("RGB")
                    images.append(img)
                    labels.append(class_dir.name)
                except Exception as e:
                    print(f"Skipping {img_file}: {e}")
    return images, labels
