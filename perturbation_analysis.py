"""
Stage 4: Perturbation Analysis
Uses YOLOv9 model to detect squirrels in original and inpainted images.
Compares detection results to measure the effect of inpainting on detection.
Computes detection drop, confidence changes, and removal rate.
"""

import os
import json
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd


def run_yolo_detection(
    model_path: str, image_paths: list, conf_threshold: float = 0.25
):
    """
    Run YOLOv9 detection on a list of images.

    Args:
        model_path: Path to YOLOv9 model (best.pt)
        image_paths: List of image paths to process
        conf_threshold: Confidence threshold for detections

    Returns:
        Dictionary mapping image paths to detection results:
        {
            'num_detections': int,
            'avg_confidence': float,
            'max_confidence': float,
            'detections': list of detection dicts
        }
    """
    # Load YOLOv9 model
    model = YOLO(model_path)

    results_dict = {}

    for img_path in tqdm(image_paths, desc="Running YOLO detection"):
        # Run inference
        results = model(img_path, conf=conf_threshold, verbose=False)

        # Extract detection information
        detections = []
        confidences = []

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]

                    confidences.append(confidence)
                    detections.append(
                        {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name,
                        }
                    )

        # Calculate statistics
        num_detections = len(detections)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        max_confidence = max(confidences) if confidences else 0.0

        results_dict[img_path] = {
            "num_detections": num_detections,
            "avg_confidence": avg_confidence,
            "max_confidence": max_confidence,
            "detections": detections,
        }

    return results_dict


def analyze_perturbations(
    detections_json: str,
    original_images_dir: str,
    inpainted_paths_json: str,
    yolo_model_path: str,
    output_dir: str,
    conf_threshold: float = 0.25,
):
    """
    Analyze perturbations: compare YOLOv9 detections on original vs inpainted images.

    Args:
        detections_json: Path to original detections.json (from Stage 1)
        original_images_dir: Directory with original images
        inpainted_paths_json: Path to inpainted_paths.json
        yolo_model_path: Path to YOLOv9 model (best.pt)
        output_dir: Directory to save analysis results
        conf_threshold: Confidence threshold for YOLO detections

    Returns:
        Tuple of (DataFrame with results, summary dictionary)
    """
    print("Analyzing perturbations using YOLOv9 detection...")

    # Load original detections (from Stage 1)
    with open(detections_json, "r") as f:
        original_detections = json.load(f)

    # Load inpainted paths
    with open(inpainted_paths_json, "r") as f:
        inpainted_paths = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Prepare image paths
    original_paths = []
    inpainted_paths_list = []
    image_names = []

    for img_name in original_detections.keys():
        original_path = os.path.join(original_images_dir, img_name)
        if img_name in inpainted_paths and os.path.exists(original_path):
            if os.path.exists(inpainted_paths[img_name]):
                original_paths.append(original_path)
                inpainted_paths_list.append(inpainted_paths[img_name])
                image_names.append(img_name)

    print(f"Processing {len(original_paths)} image pairs...")

    # Run YOLOv9 detection on original images (we already have this, but let's verify)
    print("Verifying original image detections...")
    orig_detection_results = run_yolo_detection(
        yolo_model_path, original_paths, conf_threshold
    )

    # Run YOLOv9 detection on inpainted images
    print("Running YOLOv9 detection on inpainted images...")
    inpainted_detection_results = run_yolo_detection(
        yolo_model_path, inpainted_paths_list, conf_threshold
    )

    # Compute metrics by comparing original vs inpainted detections
    results = []
    for i, img_name in enumerate(image_names):
        orig_path = original_paths[i]
        inpainted_path = inpainted_paths_list[i]

        # Get detection results
        orig_results = orig_detection_results[orig_path]
        inpainted_results = inpainted_detection_results[inpainted_path]

        # Original detections (from Stage 1)
        orig_num_detections = original_detections[img_name]["num_detections"]
        orig_avg_conf = orig_results["avg_confidence"]
        orig_max_conf = orig_results["max_confidence"]

        # Inpainted detections
        inpainted_num_detections = inpainted_results["num_detections"]
        inpainted_avg_conf = inpainted_results["avg_confidence"]
        inpainted_max_conf = inpainted_results["max_confidence"]

        # Calculate metrics
        detection_drop = orig_num_detections - inpainted_num_detections
        detection_removal_rate = (
            (detection_drop / orig_num_detections) if orig_num_detections > 0 else 0.0
        )
        avg_confidence_drop = orig_avg_conf - inpainted_avg_conf
        max_confidence_drop = orig_max_conf - inpainted_max_conf

        # Detection removed (1 if all detections removed, 0 otherwise)
        all_removed = (
            1 if inpainted_num_detections == 0 and orig_num_detections > 0 else 0
        )

        # Partial removal (1 if some but not all detections removed)
        partial_removal = (
            1 if detection_drop > 0 and inpainted_num_detections > 0 else 0
        )

        results.append(
            {
                "image_name": img_name,
                "original_num_detections": orig_num_detections,
                "inpainted_num_detections": inpainted_num_detections,
                "detection_drop": detection_drop,
                "detection_removal_rate": detection_removal_rate,
                "original_avg_confidence": orig_avg_conf,
                "inpainted_avg_confidence": inpainted_avg_conf,
                "avg_confidence_drop": avg_confidence_drop,
                "original_max_confidence": orig_max_conf,
                "inpainted_max_confidence": inpainted_max_conf,
                "max_confidence_drop": max_confidence_drop,
                "all_detections_removed": all_removed,
                "partial_removal": partial_removal,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    results_path = os.path.join(output_dir, "perturbation_analysis.csv")
    df.to_csv(results_path, index=False)

    # Summary statistics
    summary = {
        "total_images": len(results),
        "mean_detection_drop": float(df["detection_drop"].mean()),
        "std_detection_drop": float(df["detection_drop"].std()),
        "mean_detection_removal_rate": float(df["detection_removal_rate"].mean()),
        "mean_avg_confidence_drop": float(df["avg_confidence_drop"].mean()),
        "mean_max_confidence_drop": float(df["max_confidence_drop"].mean()),
        "total_all_removed": int(df["all_detections_removed"].sum()),
        "FLIP_RATE": float(df["all_detections_removed"].mean()),
        "total_partial_removal": int(df["partial_removal"].sum()),
        "partial_removal_rate": float(df["partial_removal"].mean()),
        "mean_original_detections": float(df["original_num_detections"].mean()),
        "mean_inpainted_detections": float(df["inpainted_num_detections"].mean()),
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Mean detection drop: {summary['mean_detection_drop']:.2f}")
    print(f"Mean detection removal rate: {summary['mean_detection_removal_rate']:.2%}")
    print(
        f"All detections removed: {summary['total_all_removed']}/{summary['total_images']} ({summary['FLIP_RATE']:.2%})"
    )
    print(f"Mean avg confidence drop: {summary['mean_avg_confidence_drop']:.4f}")

    return df, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perturbation Analysis using YOLOv9")
    parser.add_argument(
        "--detections",
        type=str,
        default="outputs/detections/detections.json",
        help="Path to original detections.json (from Stage 1)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="squirrel_data/test/images",
        help="Directory with original test images",
    )
    parser.add_argument(
        "--inpainted",
        type=str,
        default="outputs/inpainted/stable_diffusion/inpainted_paths.json",
        help="Path to inpainted_paths.json",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="model/best.pt",
        help="Path to YOLOv9 model (best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detections",
    )

    args = parser.parse_args()

    analyze_perturbations(
        args.detections,
        args.images,
        args.inpainted,
        args.yolo_model,
        args.output,
        conf_threshold=args.conf,
    )
