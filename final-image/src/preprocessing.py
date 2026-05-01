"""
STEP 2 - IMAGE PREPROCESSING (FIXED VERSION)
=============================================
Cleans and normalizes plate images.
CRITICAL: Uses exact same parameters as predict.py for consistency.

How to run:
  python preprocessing.py
"""

import cv2
import numpy as np
import os
from glob import glob

INPUT_FOLDER   = "augmented"
OUTPUT_FOLDER  = "preprocessed"
PREVIEW_FOLDER = "preview_check"
TARGET_WIDTH   = 448  # 2x bigger for sharper characters
TARGET_HEIGHT  = 192  # 2x bigger for sharper characters
SAVE_PREVIEWS  = True

os.makedirs(OUTPUT_FOLDER,  exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)


def preprocess_one_image(img):
    """Adaptive threshold - works better for varied lighting on real plates."""
    # Resize
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Adaptive threshold - handles different lighting across the plate
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )

    return binary


def make_preview(original_img, processed_img):
    orig_resized = cv2.resize(original_img, (TARGET_WIDTH, TARGET_HEIGHT))
    proc_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

    orig_labeled = orig_resized.copy()
    cv2.putText(orig_labeled, "ORIGINAL", (3, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    proc_labeled = proc_bgr.copy()
    cv2.putText(proc_labeled, "PROCESSED", (3, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    sep = np.zeros((TARGET_HEIGHT, 3, 3), dtype=np.uint8)
    sep[:] = (0, 200, 200)

    preview = np.hstack([orig_labeled, sep, proc_labeled])
    h, w = preview.shape[:2]
    return cv2.resize(preview, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)


def run_preprocessing():
    image_paths = (
        glob(os.path.join(INPUT_FOLDER, "*.png")) +
        glob(os.path.join(INPUT_FOLDER, "*.jpg"))
    )

    if not image_paths:
        print(f"ERROR: No images found in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(image_paths)} images")
    print(f"Saving to: {OUTPUT_FOLDER}\n")

    preview_count = 0
    MAX_PREVIEWS = 30

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        processed = preprocess_one_image(img)
        out_name = os.path.basename(path)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), processed)

        if SAVE_PREVIEWS and preview_count < MAX_PREVIEWS:
            preview = make_preview(img, processed)
            cv2.imwrite(os.path.join(PREVIEW_FOLDER, out_name), preview)
            preview_count += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(image_paths)}...")

    print(f"\n✅ Done! Preprocessed {len(image_paths)} images")
    print(f"Check '{PREVIEW_FOLDER}' for quality control")


if __name__ == "__main__":
    run_preprocessing()