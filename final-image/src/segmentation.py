"""
STEP 3 - CHARACTER SEGMENTATION (FIXED VERSION)
================================================
Extracts individual characters from preprocessed plates.
Improved character detection with better filtering.

How to run:
  python segmentation.py
"""

import cv2
import numpy as np
import os
from glob import glob

INPUT_FOLDER  = "preprocessed"
OUTPUT_FOLDER = "characters"
CHAR_SIZE     = (64, 64)  # Increased from 32x32 for better clarity

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def crop_main_row(binary):
    """Crop to ONLY the main character row, removing top decorations and bottom border."""
    h = binary.shape[0]
    # Remove top 30% (decorations, borders) and bottom 10% (plate edge)
    return binary[int(h * 0.30):int(h * 0.90), :]


def is_likely_character(x, y, w, h, img_h, img_w):
    """Filter out noise, borders, and non-character contours."""
    # Minimum size - reject tiny noise
    if w < 8 or h < 10:
        return False
    
    # Reject boxes touching edges (plate border artifacts)
    if x < 3 or (x + w) > img_w - 3:
        return False
    
    # Reject at top or bottom edges
    if y < 2 or (y + h) > img_h - 2:
        return False
    
    # Maximum width filter - allow larger characters
    if w > img_w * 0.45:
        return False
    
    # Height filters - more relaxed
    if h < img_h * 0.20:  # Too short
        return False
    if h > img_h * 0.99:  # Too tall
        return False
    
    # Aspect ratio filters
    aspect = w / h
    if aspect > 2.5:
        return False
    if aspect < 0.15:
        return False
    
    return True


def split_wide_contour(binary_row, x, y, w, h, threshold=1.2):
    """
    Split merged characters. More aggressive now.
    Recursively splits until all parts are narrow enough.
    """
    aspect = w / h
    
    # If aspect ratio is normal, keep as is
    if aspect <= threshold:
        return [(x, y, w, h)]
    
    # Extract region
    region = binary_row[y:y+h, x:x+w]
    col_sums = region.sum(axis=0)
    
    # Find the minimum point in the middle 60% (wider search area)
    search_start = int(w * 0.2)
    search_end = int(w * 0.8)
    search_sums = col_sums[search_start:search_end]
    
    if len(search_sums) == 0:
        return [(x, y, w, h)]
    
    # Find the THINNEST point (likely split location)
    split_col = search_start + int(np.argmin(search_sums))
    
    # Check if this is a valid split point (low pixel density)
    min_val = col_sums[split_col]
    mean_val = col_sums.mean()
    
    # More lenient split threshold for aggressive splitting
    if min_val < mean_val * 0.6:
        left_box = (x, y, split_col, h)
        right_box = (x + split_col, y, w - split_col, h)
        
        # RECURSIVELY split each half if still too wide
        left_splits = split_wide_contour(binary_row, *left_box, threshold)
        right_splits = split_wide_contour(binary_row, *right_box, threshold)
        
        return left_splits + right_splits
    
    # If no good split found, try to split in half anyway for very wide contours
    if aspect > 1.8:
        mid = w // 2
        left_box = (x, y, mid, h)
        right_box = (x + mid, y, w - mid, h)
        return [left_box, right_box]
    
    return [(x, y, w, h)]


def extract_character(binary_row, box, padding=10):
    """Extract single character with MAXIMUM enhancement and sharpening."""
    x, y, w, h = box
    img_h, img_w = binary_row.shape
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    char_img = binary_row[y1:y2, x1:x2]
    
    if char_img.size == 0:
        return np.zeros(CHAR_SIZE, dtype=np.uint8)
    
    # Step 1: Resize to 4x target size (256x256) for super-resolution effect
    super_size = (CHAR_SIZE[0] * 4, CHAR_SIZE[1] * 4)
    super_res = cv2.resize(char_img, super_size, interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Apply UNSHARP MASK for sharpening
    gaussian = cv2.GaussianBlur(super_res, (0, 0), 3.0)
    sharpened = cv2.addWeighted(super_res, 2.0, gaussian, -1.0, 0)
    
    # Step 3: Threshold to keep binary
    _, thresh = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    
    # Step 4: Resize down to final size with anti-aliasing
    final = cv2.resize(thresh, CHAR_SIZE, interpolation=cv2.INTER_AREA)
    
    # Step 5: Final threshold to ensure crisp black/white
    _, crisp = cv2.threshold(final, 127, 255, cv2.THRESH_BINARY)
    
    return crisp


def segment_plate(binary, plate_name, out_folder):
    row = crop_main_row(binary)
    img_h, img_w = row.shape

    contours, _ = cv2.findContours(row, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if is_likely_character(x, y, w, h, img_h, img_w):
            raw_boxes.append((x, y, w, h))

    final_boxes = []
    for box in raw_boxes:
        splits = split_wide_contour(row, *box)
        for s in splits:
            x, y, w, h = s
            if is_likely_character(x, y, w, h, img_h, img_w):
                final_boxes.append(s)

    final_boxes.sort(key=lambda b: b[0])
    
    # FILTER: Remove boxes that overlap or are too close
    filtered_boxes = []
    for i, box in enumerate(final_boxes):
        if i == 0:
            filtered_boxes.append(box)
        else:
            prev_box = filtered_boxes[-1]
            prev_x2 = prev_box[0] + prev_box[2]
            curr_x1 = box[0]
            gap = curr_x1 - prev_x2
            
            # Only keep if gap is at least 3 pixels
            if gap >= 3:
                filtered_boxes.append(box)

    for idx, box in enumerate(filtered_boxes):
        char_img = extract_character(row, box)
        out_name = f"{plate_name}_char{idx:02d}.png"
        cv2.imwrite(os.path.join(out_folder, out_name), char_img)

    return len(filtered_boxes)


def run_segmentation():
    image_paths = (
        glob(os.path.join(INPUT_FOLDER, "*.png")) +
        glob(os.path.join(INPUT_FOLDER, "*.jpg"))
    )

    if not image_paths:
        print(f"ERROR: No images in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(image_paths)} preprocessed plates")
    print(f"Saving characters to: {OUTPUT_FOLDER}\n")

    total_chars = 0
    no_chars = 0

    for i, path in enumerate(image_paths):
        binary = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if binary is None:
            continue
        
        plate_name = os.path.splitext(os.path.basename(path))[0]
        n = segment_plate(binary, plate_name, OUTPUT_FOLDER)
        total_chars += n
        
        if n == 0:
            no_chars += 1
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(image_paths)}...")

    total_plates = len(image_paths)
    success = total_plates - no_chars
    
    print(f"\n✅ Done!")
    print(f"Plates processed: {total_plates}")
    print(f"Success: {success} ({success/total_plates*100:.1f}%)")
    print(f"No chars: {no_chars} ({no_chars/total_plates*100:.1f}%)")
    print(f"Total characters: {total_chars}")
    if success > 0:
        print(f"Avg per plate: {total_chars/success:.1f}")


if __name__ == "__main__":
    run_segmentation()