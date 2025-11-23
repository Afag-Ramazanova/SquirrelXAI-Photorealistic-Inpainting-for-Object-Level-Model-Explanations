"""
Stage 5: Explainability Visualizations
Creates side-by-side visualizations showing original, detections, masks, inpainted images,
and confidence plots.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def draw_bboxes(image, detections, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image.

    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries with 'bbox' and 'confidence'
        color: Bounding box color
        thickness: Line thickness

    Returns:
        Image with bounding boxes drawn
    """
    img_with_boxes = image.copy()

    for det in detections:
        bbox = det["bbox"]
        conf = det["confidence"]
        x1, y1, x2, y2 = map(int, bbox)

        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

        # Draw confidence label
        label = f"{det['class_name']}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - label_size[1] - 5),
            (x1 + label_size[0], y1),
            color,
            -1,
        )
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    return img_with_boxes


def create_visualization_grid(
    original_image_path: str,
    detections: list,
    mask_path: str,
    inpainted_image_path: str,
    original_avg_confidence: float,
    inpainted_avg_confidence: float,
    avg_confidence_drop: float,
    all_detections_removed: int,
    original_num_detections: int,
    inpainted_num_detections: int,
    detection_drop: int,
    output_path: str,
):
    """
    Create a side by side visualization grid.

    Args:
        original_image_path: Path to original image
        detections: List of detections
        mask_path: Path to mask
        inpainted_image_path: Path to inpainted image
        original_avg_confidence: Original average detection confidence
        inpainted_avg_confidence: Inpainted average detection confidence
        avg_confidence_drop: Average confidence drop value
        all_detections_removed: Indicator if all detections were removed (0 or 1)
        original_num_detections: Number of detections in original image
        inpainted_num_detections: Number of detections in inpainted image
        detection_drop: Number of detections removed
        output_path: Path to save visualization
    """
    # Load images
    original = cv2.imread(original_image_path)
    if original is None:
        print(f"Warning: Could not load original image {original_image_path}")
        return

    mask = (
        cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_path and os.path.exists(mask_path)
        else None
    )
    inpainted = (
        cv2.imread(inpainted_image_path)
        if inpainted_image_path and os.path.exists(inpainted_image_path)
        else None
    )

    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    inpainted_rgb = (
        cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB) if inpainted is not None else None
    )

    # Draw bounding boxes on original
    original_with_boxes = draw_bboxes(original, detections)
    original_with_boxes_rgb = cv2.cvtColor(original_with_boxes, cv2.COLOR_BGR2RGB)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_rgb)
    ax1.set_title("Original Image", fontsize=14, fontweight="bold")
    ax1.axis("off")

    # Original with detections
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(original_with_boxes_rgb)
    ax2.set_title(
        f"YOLO Detections ({len(detections)} objects)", fontsize=14, fontweight="bold"
    )
    ax2.axis("off")

    # Mask
    ax3 = fig.add_subplot(gs[0, 2])
    if mask is not None:
        ax3.imshow(mask, cmap="gray")
        ax3.set_title("Segmentation Mask", fontsize=14, fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "Mask not available", ha="center", va="center", fontsize=12)
        ax3.set_title("Segmentation Mask", fontsize=14, fontweight="bold")
    ax3.axis("off")

    # Inpainted image
    ax4 = fig.add_subplot(gs[1, 0])
    if inpainted_rgb is not None:
        ax4.imshow(inpainted_rgb)
        ax4.set_title("Inpainted Image", fontsize=14, fontweight="bold")
    else:
        ax4.text(
            0.5,
            0.5,
            "Inpainted image not available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax4.set_title("Inpainted Image", fontsize=14, fontweight="bold")
    ax4.axis("off")

    # Confidence comparison
    ax5 = fig.add_subplot(gs[1, 1])
    categories = ["Original", "Inpainted"]
    confidences = [original_avg_confidence, inpainted_avg_confidence]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax5.bar(
        categories, confidences, color=colors, alpha=0.7, edgecolor="black", linewidth=2
    )
    ax5.set_ylabel("Average Confidence Score", fontsize=12, fontweight="bold")
    ax5.set_title("Detection Confidence Comparison", fontsize=14, fontweight="bold")
    ax5.set_ylim([0, 1])
    ax5.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{conf:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Metrics summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    metrics_text = f"""
    PERTURBATION METRICS
    
    Detection Drop:
    {detection_drop} detections
    
    All Removed:
    {'YES' if all_detections_removed == 1 else 'NO'}
    
    Avg Confidence Drop (Δ):
    {avg_confidence_drop:.4f}
    
    Original Avg Confidence:
    {original_avg_confidence:.4f}
    
    Inpainted Avg Confidence:
    {inpainted_avg_confidence:.4f}
    
    Original Detections: {original_num_detections}
    Inpainted Detections: {inpainted_num_detections}
    """

    ax6.text(
        0.1,
        0.5,
        metrics_text,
        fontsize=12,
        fontweight="bold",
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Overall title
    img_name = os.path.basename(original_image_path)
    fig.suptitle(
        f"Explainability Analysis: {img_name}", fontsize=16, fontweight="bold", y=0.98
    )

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_summary_plots(analysis_csv: str, output_dir: str):
    """
    Create summary plots from perturbation analysis.

    Args:
        analysis_csv: Path to perturbation_analysis.csv
        output_dir: Directory to save plots
    """
    df = pd.read_csv(analysis_csv)

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Detection Drop Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["detection_drop"], bins=30, edgecolor="black", alpha=0.7, color="skyblue"
    )
    plt.xlabel("Detection Drop (number of detections)", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.title("Distribution of Detection Drops", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.axvline(
        df["detection_drop"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["detection_drop"].mean():.2f}',
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_drop_distribution.png"), dpi=150)
    plt.close()

    # Plot 2: Original vs Inpainted Confidence Scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(
        df["original_avg_confidence"],
        df["inpainted_avg_confidence"],
        alpha=0.6,
        s=50,
        c=df["all_detections_removed"],
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=0.5,
    )
    plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="No change line")
    plt.xlabel("Original Avg Confidence", fontsize=12, fontweight="bold")
    plt.ylabel("Inpainted Avg Confidence", fontsize=12, fontweight="bold")
    plt.title(
        "Original vs Inpainted Average Confidence", fontsize=14, fontweight="bold"
    )
    plt.colorbar(label="All Removed (1=yes, 0=no)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_scatter.png"), dpi=150)
    plt.close()

    # Plot 3: Confidence Drop vs Number of Detections
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["original_num_detections"],
        df["avg_confidence_drop"],
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.xlabel("Original Number of Detections", fontsize=12, fontweight="bold")
    plt.ylabel("Avg Confidence Drop (Δ)", fontsize=12, fontweight="bold")
    plt.title("Confidence Drop vs Number of Detections", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_drop_vs_detections.png"), dpi=150)
    plt.close()

    # Plot 4: All Removed Rate Summary (equivalent to old flip rate)
    all_removed_rate = df["all_detections_removed"].mean()
    plt.figure(figsize=(8, 6))
    categories = ["Not All Removed", "All Removed"]
    values = [1 - all_removed_rate, all_removed_rate]
    colors = ["#2ecc71", "#e74c3c"]
    plt.bar(categories, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    plt.ylabel("Proportion", fontsize=12, fontweight="bold")
    plt.title(
        f"All Detections Removed Rate: {all_removed_rate:.2%}",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylim([0, 1])
    for i, v in enumerate(values):
        plt.text(
            i,
            v + 0.01,
            f"{v:.2%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_removed_rate_summary.png"), dpi=150)
    plt.close()

    # Plot 5: Detection Removal Rate Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["detection_removal_rate"],
        bins=30,
        edgecolor="black",
        alpha=0.7,
        color="orange",
    )
    plt.xlabel("Detection Removal Rate", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.title("Distribution of Detection Removal Rates", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.axvline(
        df["detection_removal_rate"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["detection_removal_rate"].mean():.2%}',
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "detection_removal_rate_distribution.png"), dpi=150
    )
    plt.close()

    print(f"Summary plots saved to {output_dir}")


def generate_all_visualizations(
    detections_json: str,
    mask_paths_json: str,
    inpainted_paths_json: str,
    analysis_csv: str,
    original_images_dir: str,
    output_dir: str,
):
    """
    Generate all visualizations for the explainability pipeline.

    Args:
        detections_json: Path to detections.json
        mask_paths_json: Path to mask_paths.json
        inpainted_paths_json: Path to inpainted_paths.json
        analysis_csv: Path to perturbation_analysis.csv
        original_images_dir: Directory with original images
        output_dir: Directory to save visualizations
    """
    print("Generating visualizations...")

    # Load data
    with open(detections_json, "r") as f:
        detections = json.load(f)

    with open(mask_paths_json, "r") as f:
        mask_paths = json.load(f)

    with open(inpainted_paths_json, "r") as f:
        inpainted_paths = json.load(f)

    df = pd.read_csv(analysis_csv)

    # Create output directories
    individual_dir = os.path.join(output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    # Generate individual visualizations
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating visualizations"):
        img_name = row["image_name"]

        if (
            img_name not in detections
            or img_name not in mask_paths
            or img_name not in inpainted_paths
        ):
            continue

        original_path = os.path.join(original_images_dir, img_name)
        mask_path = mask_paths[img_name]
        inpainted_path = inpainted_paths[img_name]

        output_path = os.path.join(
            individual_dir, f"{img_name.rsplit('.', 1)[0]}_visualization.png"
        )

        create_visualization_grid(
            original_path,
            detections[img_name]["detections"],
            mask_path,
            inpainted_path,
            row["original_avg_confidence"],
            row["inpainted_avg_confidence"],
            row["avg_confidence_drop"],
            row["all_detections_removed"],
            row["original_num_detections"],
            row["inpainted_num_detections"],
            row["detection_drop"],
            output_path,
        )

    # Generate summary plots
    create_summary_plots(analysis_csv, output_dir)

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Explainability Visualizations"
    )
    parser.add_argument(
        "--detections",
        type=str,
        default="outputs/detections/detections.json",
        help="Path to detections.json",
    )
    parser.add_argument(
        "--masks",
        type=str,
        default="outputs/masks/mask_paths.json",
        help="Path to mask_paths.json",
    )
    parser.add_argument(
        "--inpainted",
        type=str,
        default="outputs/inpainted/stable_diffusion/inpainted_paths.json",
        help="Path to inpainted_paths.json",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="outputs/analysis/perturbation_analysis.csv",
        help="Path to perturbation_analysis.csv",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="squirrel_data/test/images",
        help="Directory with original images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/final_visuals",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    generate_all_visualizations(
        args.detections,
        args.masks,
        args.inpainted,
        args.analysis,
        args.images,
        args.output,
    )
