"""
Stage 2: Mask Generation
Generates segmentation masks from bounding boxes or uses existing masks.
Supports bounding-box masks and optional SAM segmentation.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def bbox_to_mask(bbox: List[float], img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding box to binary mask.

    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        img_shape: (height, width) of the image

    Returns:
        Binary mask as numpy array
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1, x2, y2 = map(int, bbox)
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    mask[y1:y2, x1:x2] = 255
    return mask


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Combine multiple masks into a single mask.

    Args:
        masks: List of binary masks

    Returns:
        Combined binary mask
    """
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)

    combined = np.zeros_like(masks[0])
    for mask in masks:
        combined = np.maximum(combined, mask)

    return combined


def generate_masks_from_detections(
    detections_json: str,
    images_dir: str,
    output_dir: str,
    use_existing_masks: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate masks from detection results.

    Args:
        detections_json: Path to detections.json file
        images_dir: Directory containing test images
        output_dir: Directory to save generated masks
        use_existing_masks: Optional directory with existing masks to use

    Returns:
        Dictionary mapping image names to mask paths
    """
    # Load detections
    with open(detections_json, "r") as f:
        detections = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    mask_paths = {}

    # Process each image
    for img_name, det_data in tqdm(detections.items(), desc="Generating masks"):
        img_path = os.path.join(images_dir, img_name)

        # Check if we should use existing mask
        if use_existing_masks:
            # Try to find existing mask
            mask_name = img_name.rsplit(".", 1)[0] + "_mask.png"
            existing_mask_path = os.path.join(use_existing_masks, mask_name)

            if os.path.exists(existing_mask_path):
                # Copy existing mask
                output_mask_path = os.path.join(output_dir, mask_name)
                import shutil

                shutil.copy2(existing_mask_path, output_mask_path)
                mask_paths[img_name] = output_mask_path
                continue

        # Load image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        h, w = img.shape[:2]

        # Generate mask from bounding boxes
        masks = []
        for det in det_data["detections"]:
            bbox = det["bbox"]
            mask = bbox_to_mask(bbox, (h, w))
            masks.append(mask)

        # Combine all masks
        if masks:
            combined_mask = combine_masks(masks)
        else:
            # No detections - create empty mask
            combined_mask = np.zeros((h, w), dtype=np.uint8)

        # Save mask
        mask_name = img_name.rsplit(".", 1)[0] + "_mask.png"
        output_mask_path = os.path.join(output_dir, mask_name)
        cv2.imwrite(output_mask_path, combined_mask)
        mask_paths[img_name] = output_mask_path

    # Save mask paths mapping
    mapping_path = os.path.join(output_dir, "mask_paths.json")
    with open(mapping_path, "w") as f:
        json.dump(mask_paths, f, indent=2)

    print(f"\nMask generation complete! Masks saved to {output_dir}")
    print(f"Total masks generated: {len(mask_paths)}")

    return mask_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Masks from Detections")
    parser.add_argument(
        "--detections",
        type=str,
        default="outputs/detections/detections.json",
        help="Path to detections.json file",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="squirrel_data/test/images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/masks", help="Output directory for masks"
    )
    parser.add_argument(
        "--use-existing",
        type=str,
        default=None,
        help="Optional: Directory with existing masks to use (e.g., masks2/)",
    )

    args = parser.parse_args()

    generate_masks_from_detections(
        detections_json=args.detections,
        images_dir=args.images,
        output_dir=args.output,
        use_existing_masks=args.use_existing,
    )
