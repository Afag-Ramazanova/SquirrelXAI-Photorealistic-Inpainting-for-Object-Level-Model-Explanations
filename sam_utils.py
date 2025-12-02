"""
Utility functions for using Segment Anything (SAM) to generate
segmentation masks from bounding boxes.
"""

import os
from typing import List

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


def get_device() -> str:
    """Pick the best available device for SAM."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam_predictor(checkpoint_path: str, model_type: str = "vit_b") -> SamPredictor:
    """
    Load a SAM model and return a SamPredictor instance.

    Args:
        checkpoint_path: Path to SAM checkpoint (.pth)
        model_type: SAM vit_b

    Returns:
        Initialized SamPredictor
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found at: {checkpoint_path}\n"
            "Download a checkpoint (e.g. sam_vit_b.pth) and update the path."
        )

    device = get_device()
    print(f"[SAM] Loading model '{model_type}' on device: {device}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def mask_from_bbox_sam(
    predictor: SamPredictor,
    img_bgr: np.ndarray,
    bbox: List[float],
) -> np.ndarray:
    """
    Use SAM to produce a segmentation mask for a given bounding box.

    Args:
        predictor: Initialized SamPredictor
        img_bgr: Image as BGR numpy array (from cv2.imread)
        bbox: [x1, y1, x2, y2] in pixel coordinates

    Returns:
        Binary mask (uint8 array with values {0, 255})
    """
    # SAM expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    box = np.array(bbox, dtype=np.float32)

    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True,
    )

    # Pick highest-scoring mask
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8) * 255  # {0,1} -> {0,255}
    return mask
