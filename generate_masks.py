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

# NEW: SAM imports
from sam_utils import load_sam_predictor  # you already have this file


def bbox_to_mask(bbox: List[float], img_shape: Tuple[int, int]) -> np.ndarray:
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    mask[y1:y2, x1:x2] = 255
    return mask


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
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
    device: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate masks from detection results.

    Args:
        detections_json: Path to detections.json file
        images_dir: Directory containing test images
        output_dir: Directory to save generated masks
        use_existing_masks: Optional directory with existing masks to use
        use_sam: If True, use SAM to refine bboxes into segmentation masks
        sam_checkpoint: Path to SAM checkpoint (.pth)
        sam_model_type: SAM model type (e.g., 'vit_b')
        device: torch device ('cuda', 'mps', 'cpu'); if None, sam_utils picks

    Returns:
        Dictionary mapping image names to mask paths
    """
    with open(detections_json, "r") as f:
        detections = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # --- Load SAM predictor once (if requested) ---
    sam_predictor = None
    if use_sam:
        if sam_checkpoint is None:
            raise ValueError("use_sam=True but sam_checkpoint is None")
        print(f"[SAM] Loading model '{sam_model_type}' from {sam_checkpoint}")
        sam_predictor = load_sam_predictor(
            checkpoint_path=sam_checkpoint,
            model_type=sam_model_type,
            device=device,
        )

    mask_paths: Dict[str, str] = {}

    for img_name, det_data in tqdm(detections.items(), desc="Generating masks"):
        img_path = os.path.join(images_dir, img_name)

        # 1) Use existing high-quality masks if provided
        if use_existing_masks:
            mask_name = img_name.rsplit(".", 1)[0] + "_mask.png"
            existing_mask_path = os.path.join(use_existing_masks, mask_name)
            if os.path.exists(existing_mask_path):
                output_mask_path = os.path.join(output_dir, mask_name)
                import shutil

                shutil.copy2(existing_mask_path, output_mask_path)
                mask_paths[img_name] = output_mask_path
                continue

        # 2) Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        h, w = img_bgr.shape[:2]

        # 3) Build masks
        per_det_masks = []

        if use_sam and sam_predictor is not None and det_data["num_detections"] > 0:
            # SAM expects RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(img_rgb)

            for det in det_data["detections"]:
                x1, y1, x2, y2 = det["bbox"]

                # Center point of bbox as foreground prompt
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                point_coords = np.array([[cx, cy]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

                # Optional: give SAM the box as a hint as well
                box = np.array([[x1, y1, x2, y2]], dtype=np.float32)

                sam_masks, scores, logits = sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=False,
                )
                # sam_masks[0] is boolean mask (H,W)
                m = sam_masks[0].astype(np.uint8) * 255
                per_det_masks.append(m)

        else:
            # Fallback: simple bbox-to-mask like before
            for det in det_data["detections"]:
                bbox = det["bbox"]
                m = bbox_to_mask(bbox, (h, w))
                per_det_masks.append(m)

        # 4) Combine all object masks for this image
        if per_det_masks:
            combined_mask = combine_masks(per_det_masks)
        else:
            combined_mask = np.zeros((h, w), dtype=np.uint8)

        # 5) Save
        mask_name = img_name.rsplit(".", 1)[0] + "_mask.png"
        output_mask_path = os.path.join(output_dir, mask_name)
        cv2.imwrite(output_mask_path, combined_mask)
        mask_paths[img_name] = output_mask_path

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
        "--output",
        type=str,
        default="outputs/masks",
        help="Output directory for masks",
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
        help="Use SAM instead of plain bounding boxes",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_b",
        help="SAM model type (e.g., vit_b, vit_l)",
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
