"""
STEP 1 - DATA AUGMENTATION
===========================
Takes your original plate images and creates augmented versions.
Turns 2070 images into ~12,000+ images for better training.

How to run:
  python augmentation.py
"""

import cv2
import numpy as np
import os
from glob import glob

INPUT_FOLDER  = "images"
OUTPUT_FOLDER = "augmented"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def adjust_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_noise(img):
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def blur_image(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def augment_one_image(img, base_name, out_folder):
    count = 0
    img_resized = cv2.resize(img, (224, 64))
    
    cv2.imwrite(os.path.join(out_folder, f"{base_name}_orig.png"), img_resized)
    count += 1

    for angle in [-10, -5, 5, 10]:
        rotated = rotate_image(img_resized, angle)
        cv2.imwrite(os.path.join(out_folder, f"{base_name}_rot{angle}.png"), rotated)
        count += 1

    for factor in [0.6, 0.8, 1.2, 1.4]:
        bright = adjust_brightness(img_resized, factor)
        cv2.imwrite(os.path.join(out_folder, f"{base_name}_bright{int(factor*10)}.png"), bright)
        count += 1

    noisy = add_noise(img_resized)
    cv2.imwrite(os.path.join(out_folder, f"{base_name}_noise.png"), noisy)
    count += 1

    blurred = blur_image(img_resized)
    cv2.imwrite(os.path.join(out_folder, f"{base_name}_blur.png"), blurred)
    count += 1

    dark_noisy = add_noise(adjust_brightness(img_resized, 0.5))
    cv2.imwrite(os.path.join(out_folder, f"{base_name}_darknoise.png"), dark_noisy)
    count += 1

    return count


def run_augmentation():
    image_paths = (
        glob(os.path.join(INPUT_FOLDER, "*.png")) +
        glob(os.path.join(INPUT_FOLDER, "*.jpg")) +
        glob(os.path.join(INPUT_FOLDER, "*.jpeg"))
    )

    if not image_paths:
        print(f"ERROR: No images found in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(image_paths)} original images.")
    print(f"Saving to: {OUTPUT_FOLDER}\n")

    total_created = 0
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        base_name = os.path.splitext(os.path.basename(path))[0]
        count = augment_one_image(img, base_name, OUTPUT_FOLDER)
        total_created += count

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(image_paths)}...")

    print(f"\n✅ Done!")
    print(f"Original: {len(image_paths)} → Augmented: {total_created}")


if __name__ == "__main__":
    run_augmentation()