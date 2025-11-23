"""
Stage 3: Inpainting Backends
Performs object-level inpainting using Stable Diffusion via Replicate API.
Includes verification and testing functionality.
"""

import os
import json
import base64
import replicate
import requests
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import time


# Default prompt templates
DEFAULT_REMOVAL_PROMPT = "seamless natural background, consistent lighting, no animals"
DEFAULT_REPLACEMENT_PROMPT = "photorealistic object in natural setting"
DEFAULT_NEGATIVE_PROMPT = "duplicate, distortion, artifacts, low quality"


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def inpaint_with_stable_diffusion(
    image_path: str,
    mask_path: str,
    prompt: str = DEFAULT_REMOVAL_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
) -> Optional[np.ndarray]:
    """
    Perform inpainting using Stable Diffusion via Replicate API.

    Args:
        image_path: Path to input image
        mask_path: Path to binary mask
        prompt: Inpainting prompt
        negative_prompt: Negative prompt
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale

    Returns:
        Inpainted image as numpy array, or None if failed
    """
    try:
        # Convert to absolute paths
        abs_image_path = os.path.abspath(image_path)
        abs_mask_path = os.path.abspath(mask_path)

        # Verify files exist
        if not os.path.exists(abs_image_path):
            raise FileNotFoundError(f"Image file not found: {abs_image_path}")
        if not os.path.exists(abs_mask_path):
            raise FileNotFoundError(f"Mask file not found: {abs_mask_path}")

        # Get Replicate client
        client = replicate.Client()

        # Upload files to Replicate using the API
        # Files must be uploaded first, then the returned file objects are used
        # Open files and keep them open during upload (matching user's working pattern)
        with open(abs_image_path, "rb") as img_f, open(abs_mask_path, "rb") as mask_f:
            image_file_obj = client.files.create(file=img_f)
            mask_file_obj = client.files.create(file=mask_f)

        # File objects from client.files.create() return File objects with URLs
        # Extract the URL string from the File object's urls dictionary
        # File objects have a 'urls' dict with a 'get' key containing the URL
        try:
            image_input = (
                image_file_obj.urls.get("get")
                if hasattr(image_file_obj, "urls")
                else image_file_obj
            )
            mask_input = (
                mask_file_obj.urls.get("get")
                if hasattr(mask_file_obj, "urls")
                else mask_file_obj
            )
        except (AttributeError, KeyError):
            # Fallback: try using File objects directly (as in user's working code)
            image_input = image_file_obj
            mask_input = mask_file_obj

        # Use Replicate API - Stable Diffusion Inpainting
        # Pass the file URLs or File objects
        output = client.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "image": image_input,  # URL string or File object
                "mask": mask_input,  # URL string or File object
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )

        # Download the result
        # Output is typically a list containing a URL
        if isinstance(output, str):
            # If output is a URL string, download it
            result_url = output
        elif isinstance(output, list) and len(output) > 0:
            # If output is a list of URLs
            result_url = output[0]
        else:
            print(f"Unexpected output format: {type(output)}")
            return None

        # Download and convert the image
        response = requests.get(result_url)
        inpainted_image = Image.open(BytesIO(response.content)).convert("RGB")

        # Convert to numpy array
        inpainted_array = np.array(inpainted_image)

        return inpainted_array

    except Exception as e:
        error_str = str(e)
        # Re raise rate limit errors so they can be handled by retry logic
        if (
            "429" in error_str
            or "throttled" in error_str.lower()
            or "rate limit" in error_str.lower()
        ):
            raise  # Re raise to allow retry logic in batch_inpaint to handle it
        print(f"Error during inpainting: {error_str}")
        return None


def verify_inpainting_output(
    inpainted_image: np.ndarray, original_shape: Tuple[int, int]
) -> bool:
    """
    Verify that inpainting output is valid.

    Args:
        inpainted_image: Inpainted image as numpy array
        original_shape: Original image shape (height, width)

    Returns:
        True if valid, False otherwise
    """
    if inpainted_image is None:
        return False

    # Check shape
    if len(inpainted_image.shape) != 3 or inpainted_image.shape[2] != 3:
        print(f"Invalid shape: {inpainted_image.shape}")
        return False

    # Check if empty/black tensor
    if np.all(inpainted_image == 0):
        print("Image is all black")
        return False

    # Check pixel values (should be 0-255)
    if inpainted_image.min() < 0 or inpainted_image.max() > 255:
        print(
            f"Pixel values out of range: [{inpainted_image.min()}, {inpainted_image.max()}]"
        )
        return False

    # Check if image is corrupted (all same value)
    if np.std(inpainted_image) < 1.0:
        print("Image appears corrupted (low variance)")
        return False

    return True


def test_backend(
    test_image_path: str,
    test_mask_path: str,
    output_dir: str,
    backend_name: str = "stable_diffusion",
):
    """
    Test inpainting backend with a sample image.

    Args:
        test_image_path: Path to test image
        test_mask_path: Path to test mask
        output_dir: Directory to save test outputs
        backend_name: Name of the backend
    """
    print(f"\nTesting {backend_name} backend...")
    print(f"Test image: {test_image_path}")
    print(f"Test mask: {test_mask_path}")

    # Load original image to get shape
    original_img = cv2.imread(test_image_path)
    if original_img is None:
        print(f"Error: Could not load test image {test_image_path}")
        return False

    original_shape = original_img.shape[:2]

    # Perform inpainting
    print("Running inpainting...")
    inpainted = inpaint_with_stable_diffusion(
        test_image_path, test_mask_path, prompt=DEFAULT_REMOVAL_PROMPT
    )

    if inpainted is None:
        print("Inpainting failed!")
        return False

    # Verify output
    print("Verifying output...")
    is_valid = verify_inpainting_output(inpainted, original_shape)

    if not is_valid:
        print("Verification failed!")
        return False

    # Save test output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"test_output_{backend_name}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))

    print(f"✓ Backend test passed! Output saved to {output_path}")
    return True


def batch_inpaint(
    mask_paths_json: str,
    images_dir: str,
    output_dir: str,
    backend: str = "stable_diffusion",
    prompt: str = DEFAULT_REMOVAL_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    delay_between_requests: float = 12.0,
    max_retries: int = 3,
) -> Dict[str, str]:
    """
    Perform batch inpainting on all images.

    Args:
        mask_paths_json: Path to mask_paths.json file
        images_dir: Directory containing test images
        output_dir: Directory to save inpainted images
        backend: Inpainting backend to use
        prompt: Inpainting prompt
        negative_prompt: Negative prompt
        delay_between_requests: Delay in seconds between API requests (default: 12.0)
                              For accounts with <$5 credit: 6 req/min = 10s minimum, use 12s+ to be safe
        max_retries: Maximum number of retries for rate limit errors (default: 3)

    Returns:
        Dictionary mapping image names to inpainted image paths
    """
    # Load mask paths
    with open(mask_paths_json, "r") as f:
        mask_paths = json.load(f)

    # Create output directory
    backend_output_dir = os.path.join(output_dir, backend)
    os.makedirs(backend_output_dir, exist_ok=True)

    inpainted_paths = {}

    # Process each image
    for img_name, mask_path in tqdm(
        mask_paths.items(), desc=f"Inpainting with {backend}"
    ):
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found: {mask_path}")
            continue

        # Perform inpainting with retry logic for rate limiting
        inpainted = None
        retry_count = 0

        while retry_count <= max_retries:
            try:
                inpainted = inpaint_with_stable_diffusion(
                    img_path, mask_path, prompt=prompt, negative_prompt=negative_prompt
                )
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error (429)
                if (
                    "429" in error_str
                    or "throttled" in error_str.lower()
                    or "rate limit" in error_str.lower()
                ):
                    retry_count += 1
                    if retry_count <= max_retries:
                        # Exponential backoff: wait longer on each retry
                        wait_time = delay_between_requests * (2**retry_count)
                        print(
                            f"\n⚠ Rate limited. Waiting {wait_time:.1f}s before retry {retry_count}/{max_retries}..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(
                            f"\n✗ Rate limit exceeded after {max_retries} retries. Skipping {img_name}"
                        )
                        break
                else:
                    # Not a rate limit error, don't retry
                    print(f"\n✗ Error for {img_name}: {error_str}")
                    break

        if inpainted is None:
            print(f"Warning: Inpainting failed for {img_name}")
            continue

        # Save inpainted image
        output_name = img_name.rsplit(".", 1)[0] + "_inpainted.jpg"
        output_path = os.path.join(backend_output_dir, output_name)
        cv2.imwrite(output_path, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
        inpainted_paths[img_name] = output_path

        # Add delay between requests to respect rate limits
        # For accounts with <$5: 6 req/min = 10s minimum, using 12s+ to be safe
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)

    # Save paths mapping
    mapping_path = os.path.join(backend_output_dir, "inpainted_paths.json")
    with open(mapping_path, "w") as f:
        json.dump(inpainted_paths, f, indent=2)

    print(f"\nInpainting complete! Results saved to {backend_output_dir}")
    print(f"Total images inpainted: {len(inpainted_paths)}")

    return inpainted_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inpainting with Stable Diffusion")
    parser.add_argument(
        "--masks",
        type=str,
        default="outputs/masks/mask_paths.json",
        help="Path to mask_paths.json file",
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
        default="outputs/inpainted",
        help="Output directory for inpainted images",
    )
    parser.add_argument(
        "--backend", type=str, default="stable_diffusion", help="Inpainting backend"
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_REMOVAL_PROMPT, help="Inpainting prompt"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt",
    )
    parser.add_argument("--test", action="store_true", help="Run backend test first")
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Path to test image for verification",
    )
    parser.add_argument(
        "--test-mask", type=str, default=None, help="Path to test mask for verification"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=12.0,
        help="Delay in seconds between API requests (default: 12.0). For accounts with <$5 credit, use 12+ seconds to avoid rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for rate limit errors (default: 3)",
    )

    args = parser.parse_args()

    # Test backend if requested
    if args.test:
        if args.test_image and args.test_mask:
            test_backend(
                args.test_image, args.test_mask, "outputs/backend_tests", args.backend
            )
        else:
            print("Warning: --test requires --test-image and --test-mask")

    # Run batch inpainting
    batch_inpaint(
        mask_paths_json=args.masks,
        images_dir=args.images,
        output_dir=args.output,
        backend=args.backend,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        delay_between_requests=args.delay,
        max_retries=args.max_retries,
    )
