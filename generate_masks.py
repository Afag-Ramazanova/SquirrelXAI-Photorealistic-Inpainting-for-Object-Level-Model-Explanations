"""
Stage 2: Mask Generation

Generates segmentation masks from bounding boxes, existing masks, or SAM.

Supports:
- Bounding-box masks (original behavior)
- Optional use of existing masks (e.g., masks2/)
- Optional SAM-based segmentation from YOLO bounding boxes
"""

import json
import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Local utility for SAM (optional)
try:
    from sam_utils import load_sam_predictor, mask_from_bbox_sam

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


def bbox_to_mask(bbox: List[float], img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding box to binary mask.

    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        img_shape: (height, width) of the image

    Returns:
        Binary mask as numpy array (0 or 255)
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
    use_sam: bool = False,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_b",
) -> Dict[str, str]:
    """
    Generate masks from detection results.

    Args:
        detections_json: Path to detections.json file
        images_dir: Directory containing test images
        output_dir: Directory to save generated masks
        use_existing_masks: Optional directory with existing masks to use
        use_sam: If True, use SAM to get segmentation masks from YOLO bboxes
        sam_checkpoint: Path to SAM checkpoint (.pth) when use_sam=True
        sam_model_type: SAM model type key 'vit_b'

    Returns:
        Dictionary mapping image names to mask paths
    """
    # Load detections
    with open(detections_json, "r") as f:
        detections = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare SAM predictor if requested
    sam_predictor = None
    if use_sam:
        if not SAM_AVAILABLE:
            raise ImportError(
                "SAM utilities not available. "
                "Make sure segment-anything and sam_utils.py are installed."
            )
        if sam_checkpoint is None:
            raise ValueError("use_sam=True but sam_checkpoint is None.")
        sam_predictor = load_sam_predictor(
            checkpoint_path=sam_checkpoint,
            model_type=sam_model_type,
        )

    mask_paths: Dict[str, str] = {}

    # Process each image
    for img_name, det_data in tqdm(detections.items(), desc="Generating masks"):
        img_path = os.path.join(images_dir, img_name)

        # If requested, use an existing pre-made mask (e.g., masks2/)
        if use_existing_masks:
            mask_name = img_name.rsplit(".", 1)[0] + "_mask.png"
            existing_mask_path = os.path.join(use_existing_masks, mask_name)

            if os.path.exists(existing_mask_path):
                import shutil

                output_mask_path = os.path.join(output_dir, mask_name)
                shutil.copy2(existing_mask_path, output_mask_path)
                mask_paths[img_name] = output_mask_path
                continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        h, w = img.shape[:2]
        masks: List[np.ndarray] = []

        # Generate mask(s) from detections
        for det in det_data.get("detections", []):
            bbox = det["bbox"]

            if use_sam and sam_predictor is not None:
                try:
                    mask = mask_from_bbox_sam(sam_predictor, img, bbox)
                except Exception as e:
                    print(
                        f"Warning: SAM failed for {img_name} with bbox {bbox}: {e}. "
                        "Falling back to rectangular bbox mask."
                    )
                    mask = bbox_to_mask(bbox, (h, w))
            else:
                mask = bbox_to_mask(bbox, (h, w))

            masks.append(mask)

        # Combine all masks for this image
        if masks:
            combined_mask = combine_masks(masks)
        else:
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
    parser.add_argument(
        "--use-sam",
        action="store_true",
        help="Use SAM to generate segmentation masks from YOLO bounding boxes",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint (.pth) when --use-sam is set",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_b",
        help="SAM model type (e.g., vit_b, vit_l, vit_h)",
    )

    args = parser.parse_args()

    generate_masks_from_detections(
        detections_json=args.detections,
        images_dir=args.images,
        output_dir=args.output,
        use_existing_masks=args.use_existing,
        use_sam=args.use_sam,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
    )
