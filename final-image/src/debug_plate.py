"""
DEBUG TOOL - See exactly what's happening with your plate
==========================================================
This shows each step of the process so we can see WHERE it fails.

How to run:
  py -3.11 debug_plate.py images/0044_license_plate_1.png
"""

import cv2
import numpy as np
import sys
import os

TARGET_WIDTH = 448
TARGET_HEIGHT = 192


def preprocess_v1(img):
    """Simple Otsu."""
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(binary == 255) / binary.size
    print(f"  V1 white ratio: {white_ratio*100:.1f}%")
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)
        print(f"  V1: INVERTED")
    return binary, gray


def preprocess_v2(img):
    """Adaptive threshold."""
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )
    return binary


def preprocess_v3(img):
    """Otsu with inversion."""
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def show_contours(binary, label):
    """Show all contours found."""
    row = binary[int(binary.shape[0] * 0.30):int(binary.shape[0] * 0.90), :]
    contours, _ = cv2.findContours(row, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n  === {label} ===")
    print(f"  Image size: {binary.shape}")
    print(f"  Cropped row size: {row.shape}")
    print(f"  Total contours found: {len(contours)}")
    
    # Draw all contours
    debug = cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
    img_h, img_w = row.shape
    
    valid_count = 0
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        
        # Show ALL contours with their info
        color = (0, 255, 0)  # green by default
        reject_reason = None
        
        if w < 8 or h < 10:
            color = (0, 0, 255)  # red
            reject_reason = "too small"
        elif x < 3 or (x + w) > img_w - 3:
            color = (0, 0, 255)
            reject_reason = "edge"
        elif y < 2 or (y + h) > img_h - 2:
            color = (0, 0, 255)
            reject_reason = "top/bottom edge"
        elif w > img_w * 0.45:
            color = (0, 0, 255)
            reject_reason = "too wide"
        elif h < img_h * 0.20:
            color = (0, 0, 255)
            reject_reason = "too short"
        elif h > img_h * 0.99:
            color = (0, 0, 255)
            reject_reason = "too tall"
        elif aspect > 2.5:
            color = (0, 0, 255)
            reject_reason = "aspect too wide"
        elif aspect < 0.15:
            color = (0, 0, 255)
            reject_reason = "aspect too narrow"
        else:
            valid_count += 1
        
        cv2.rectangle(debug, (x, y), (x+w, y+h), color, 1)
        
        if i < 30:  # Show first 30 contours info
            status = f"REJECTED: {reject_reason}" if reject_reason else "ACCEPTED"
            print(f"    [{i}] x={x:3d} y={y:3d} w={w:3d} h={h:3d} aspect={aspect:.2f} - {status}")
    
    print(f"\n  Valid characters: {valid_count}")
    return debug


def debug_plate(image_path):
    print("="*60)
    print("  DEBUG TOOL")
    print("="*60)
    print(f"\nImage: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return
    
    print(f"Original size: {img.shape}")
    
    # Save original
    cv2.imwrite("debug_00_original.png", img)
    print("✅ Saved: debug_00_original.png")
    
    # Try V1: Simple Otsu
    print("\n--- Trying V1: Simple Otsu ---")
    v1_binary, gray = preprocess_v1(img)
    cv2.imwrite("debug_01_grayscale.png", gray)
    cv2.imwrite("debug_02_v1_otsu.png", v1_binary)
    print("✅ Saved: debug_02_v1_otsu.png")
    debug_v1 = show_contours(v1_binary, "V1: Simple Otsu")
    cv2.imwrite("debug_03_v1_contours.png", debug_v1)
    
    # Try V2: Adaptive
    print("\n--- Trying V2: Adaptive Threshold ---")
    v2_binary = preprocess_v2(img)
    cv2.imwrite("debug_04_v2_adaptive.png", v2_binary)
    print("✅ Saved: debug_04_v2_adaptive.png")
    debug_v2 = show_contours(v2_binary, "V2: Adaptive")
    cv2.imwrite("debug_05_v2_contours.png", debug_v2)
    
    # Try V3: Otsu inverted
    print("\n--- Trying V3: Otsu Inverted ---")
    v3_binary = preprocess_v3(img)
    cv2.imwrite("debug_06_v3_otsu_inv.png", v3_binary)
    print("✅ Saved: debug_06_v3_otsu_inv.png")
    debug_v3 = show_contours(v3_binary, "V3: Otsu Inverted")
    cv2.imwrite("debug_07_v3_contours.png", debug_v3)
    
    print("\n" + "="*60)
    print("✅ Debug complete! Check these files:")
    print("  debug_00_original.png     - Original")
    print("  debug_01_grayscale.png    - Grayscale")
    print("  debug_02_v1_otsu.png      - V1 preprocessing")
    print("  debug_03_v1_contours.png  - V1 contours (green=valid, red=rejected)")
    print("  debug_04_v2_adaptive.png  - V2 preprocessing")
    print("  debug_05_v2_contours.png  - V2 contours")
    print("  debug_06_v3_otsu_inv.png  - V3 preprocessing")
    print("  debug_07_v3_contours.png  - V3 contours")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.11 debug_plate.py <image_path>")
        sys.exit(1)
    debug_plate(sys.argv[1])
