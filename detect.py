"""
Stage 1: YOLOv9 Detection
Detects squirrels in test images using trained YOLOv9 model.
Outputs bounding boxes, scores, and labels in JSON format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


def detect_squirrels(
    model_path: str, images_dir: str, output_dir: str, conf_threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Run YOLOv9 detection on test images.

    Args:
        model_path: Path to best.pt model file
        images_dir: Directory containing test images
        output_dir: Directory to save detection results
        conf_threshold: Confidence threshold for detections

    Returns:
        Dictionary with detection results for all images
    """
    # Load YOLOv9 model
    print(f"Loading YOLOv9 model from {model_path}...")
    model = YOLO(model_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_files = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    print(f"Found {len(image_files)} images to process")

    all_results = {}

    # Process each image
    for img_file in tqdm(image_files, desc="Detecting squirrels"):
        img_path = os.path.join(images_dir, img_file)

        # Run inference
        results = model(img_path, conf=conf_threshold, verbose=False)

        # Extract detection information
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]

                    detections.append(
                        {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name,
                        }
                    )

        # Store results
        all_results[img_file] = {
            "image_path": img_path,
            "num_detections": len(detections),
            "detections": detections,
        }

    # Save results to JSON
    output_json = os.path.join(output_dir, "detections.json")
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetection complete! Results saved to {output_json}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total detections: {sum(r['num_detections'] for r in all_results.values())}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv9 Squirrel Detection")
    parser.add_argument(
        "--model", type=str, default="model/best.pt", help="Path to YOLOv9 model file"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="squirrel_data/test/images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/detections",
        help="Output directory for detection results",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    detect_squirrels(
        model_path=args.model,
        images_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf,
    )
