# core/yolo_trainer.py

import subprocess
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Union, List, Optional
import yaml

class YOLOTrainer:
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initialize YOLO trainer with specified model.

        Args:
            model_name (str): Name of the YOLO model to use (e.g., 'yolov8n.pt', 'yolov8s.pt')
        """
        self.model_name = model_name
        self.model = None
        self.trained_model_path = None

    def load_model(self, model_path: Optional[str] = None):
        """Load YOLO model for training or inference."""
        try:
            if model_path:
                self.model = YOLO(model_path)
                print(f"Loaded model from: {model_path}")
            else:
                self.model = YOLO(self.model_name)
                print(f"Loaded pretrained model: {self.model_name}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def train_yolo(self, data_yaml: str, model="yolov8n.pt", epochs=50, imgsz=640, 
                   project="yolo_runs", name="train", device="", batch=16, lr0=0.01, 
                   save_period=10, patience=100, **kwargs):
        """
        Train YOLOv8 using Ultralytics Python API.

        Args:
            data_yaml (str): Path to dataset YAML file
            model (str): Model architecture to use
            epochs (int): Number of training epochs
            imgsz (int): Image size for training
            project (str): Project name
            name (str): Experiment name
            device (str): Device to use for training ('', '0', '1', 'cpu')
            batch (int): Batch size
            lr0 (float): Initial learning rate
            save_period (int): Save checkpoint every n epochs
            patience (int): EarlyStopping patience
            **kwargs: Additional training arguments
        """
        try:
            # Load model
            if not self.model:
                self.load_model(model)

            # Train the model
            print(f"Starting training with:")
            print(f"  Data: {data_yaml}")
            print(f"  Model: {model}")
            print(f"  Epochs: {epochs}")
            print(f"  Image size: {imgsz}")
            print(f"  Batch size: {batch}")

            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                project=project,
                name=name,
                device=device,
                batch=batch,
                lr0=lr0,
                save_period=save_period,
                patience=patience,
                **kwargs
            )

            # Store the trained model path
            self.trained_model_path = self.model.trainer.best
            print(f"Training completed! Best model saved at: {self.trained_model_path}")

            return results

        except Exception as e:
            print(f"Training failed: {e}")
            return None

    def train_yolo_cli(self, data_yaml: str, model="yolov8n.pt", epochs=50, imgsz=640, 
                       project="yolo_runs", **kwargs):
        """
        Train YOLOv8 using Ultralytics CLI (original implementation enhanced).
        """
        try:
            cmd = [
                "yolo", "task=detect", "mode=train",
                f"data={data_yaml}",
                f"model={model}",
                f"epochs={epochs}",
                f"imgsz={imgsz}",
                f"project={project}"
            ]

            # Add additional arguments
            for key, value in kwargs.items():
                cmd.append(f"{key}={value}")

            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Training completed successfully!")
            print(result.stdout)
            return True

        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            print(f"Error output: {e.stderr}")
            return False

    def predict_image(self, image_path: Union[str, np.ndarray], conf=0.25, iou=0.7, 
                     model_path: Optional[str] = None, save_results=True, 
                     show_labels=True, show_conf=True):
        """
        Perform object detection on a single image.

        Args:
            image_path (str or np.ndarray): Path to image or image array
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            model_path (str): Path to trained model (optional)
            save_results (bool): Whether to save prediction results
            show_labels (bool): Whether to show class labels
            show_conf (bool): Whether to show confidence scores

        Returns:
            results: YOLO prediction results
        """
        try:
            # Load model if not already loaded or if different model specified
            if not self.model or model_path:
                self.load_model(model_path or self.trained_model_path or self.model_name)

            # Run prediction
            results = self.model(
                image_path,
                conf=conf,
                iou=iou,
                save=save_results,
                show_labels=show_labels,
                show_conf=show_conf
            )

            return results

        except Exception as e:
            print(f"Prediction failed: {e}")
            return None

    def predict_batch(self, image_paths: List[str], conf=0.25, iou=0.7, 
                     model_path: Optional[str] = None):
        """
        Perform object detection on multiple images.
        """
        try:
            if not self.model or model_path:
                self.load_model(model_path or self.trained_model_path or self.model_name)

            results = self.model(image_paths, conf=conf, iou=iou)
            return results

        except Exception as e:
            print(f"Batch prediction failed: {e}")
            return None

    def extract_predictions(self, results, class_names=None):
        """
        Extract prediction information from YOLO results.

        Args:
            results: YOLO prediction results
            class_names (dict): Optional mapping of class IDs to names

        Returns:
            dict: Extracted prediction information
        """
        predictions = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Extract bounding box coordinates (xyxy format)
                bboxes = boxes.xyxy.cpu().numpy()
                # Extract confidence scores
                confidences = boxes.conf.cpu().numpy()
                # Extract class IDs
                class_ids = boxes.cls.cpu().numpy().astype(int)

                # Get class names
                if class_names:
                    classes = [class_names.get(cls_id, f"class_{cls_id}") for cls_id in class_ids]
                else:
                    classes = [result.names[cls_id] for cls_id in class_ids]

                # Combine all information
                for bbox, conf, cls_id, cls_name in zip(bboxes, confidences, class_ids, classes):
                    predictions.append({
                        'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': cls_name
                    })

        return predictions

    def draw_predictions(self, image: np.ndarray, predictions: List[dict], 
                        color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes and labels on image.

        Args:
            image (np.ndarray): Input image
            predictions (list): List of prediction dictionaries
            color (tuple): Color for bounding boxes
            thickness (int): Line thickness

        Returns:
            np.ndarray: Image with drawn predictions
        """
        annotated_image = image.copy()

        for pred in predictions:
            bbox = pred['bbox']
            confidence = pred['confidence']
            class_name = pred['class_name']

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_image

    def save_model(self, save_path: str):
        """Save the trained model to specified path."""
        if self.model and self.trained_model_path:
            try:
                # Copy the best model to the specified path
                import shutil
                shutil.copy2(self.trained_model_path, save_path)
                print(f"Model saved to: {save_path}")
                return True
            except Exception as e:
                print(f"Error saving model: {e}")
                return False
        else:
            print("No trained model to save")
            return False

    def validate_model(self, data_yaml: Optional[str] = None):
        """Validate the trained model."""
        if self.model:
            try:
                metrics = self.model.val(data=data_yaml)
                return metrics
            except Exception as e:
                print(f"Validation failed: {e}")
                return None
        else:
            print("No model loaded for validation")
            return None

# Convenience functions for backward compatibility
def train_yolo(data_yaml: str, model="yolov8n.pt", epochs=50, imgsz=640, project="yolo_runs"):
    """Train YOLOv8 using Ultralytics CLI (original function)."""
    trainer = YOLOTrainer()
    return trainer.train_yolo_cli(data_yaml, model, epochs, imgsz, project)

# Example usage and utility functions
def create_sample_data_yaml(train_path: str, val_path: str, class_names: List[str], 
                           save_path: str = "data.yaml"):
    """Create a sample data.yaml file for YOLO training."""
    data_config = {
        'train': train_path,
        'val': val_path,
        'nc': len(class_names),
        'names': class_names
    }

    with open(save_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"Created data.yaml at: {save_path}")
    return save_path

if __name__ == "__main__":
    # Example usage
    trainer = YOLOTrainer("yolov8n.pt")

    # Train model (uncomment to use)
    # trainer.train_yolo("path/to/data.yaml", epochs=100)

    # Predict on image (uncomment to use)
    # results = trainer.predict_image("path/to/image.jpg")
    # predictions = trainer.extract_predictions(results)
    # print(predictions)
