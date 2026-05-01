"""
PREDICT.PY - EXACT MATCH to preprocessing.py + segmentation.py
================================================================
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import json
import sys
import os

MODEL_FILE = "model.pth1"
CLASSES_FILE = "classes.json"
IMG_SIZE = 64
TARGET_WIDTH = 448
TARGET_HEIGHT = 192


class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super(CharCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============ COPIED EXACTLY FROM preprocessing.py ============
def preprocess_one_image(img):
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


# ============ COPIED EXACTLY FROM segmentation.py ============
def crop_main_row(binary):
    h = binary.shape[0]
    return binary[int(h * 0.30):int(h * 0.90), :]


def is_likely_character(x, y, w, h, img_h, img_w):
    if w < 8 or h < 10:
        return False
    if x < 3 or (x + w) > img_w - 3:
        return False
    if y < 2 or (y + h) > img_h - 2:
        return False
    if w > img_w * 0.45:
        return False
    if h < img_h * 0.20:
        return False
    if h > img_h * 0.99:
        return False
    aspect = w / h
    if aspect > 2.5:
        return False
    if aspect < 0.30:  # Stricter aspect (was 0.15) - rejects narrow dividers
        return False
    # REJECT PLATE DIVIDERS: thin tall vertical lines
    if w < 25 and h > img_h * 0.60:
        return False
    return True


def split_wide_contour(binary_row, x, y, w, h, threshold=1.2):
    aspect = w / h
    if aspect <= threshold:
        return [(x, y, w, h)]
    region = binary_row[y:y+h, x:x+w]
    col_sums = region.sum(axis=0)
    search_start = int(w * 0.2)
    search_end = int(w * 0.8)
    search_sums = col_sums[search_start:search_end]
    if len(search_sums) == 0:
        return [(x, y, w, h)]
    split_col = search_start + int(np.argmin(search_sums))
    min_val = col_sums[split_col]
    mean_val = col_sums.mean()
    if min_val < mean_val * 0.6:
        left_box = (x, y, split_col, h)
        right_box = (x + split_col, y, w - split_col, h)
        left_splits = split_wide_contour(binary_row, *left_box, threshold)
        right_splits = split_wide_contour(binary_row, *right_box, threshold)
        return left_splits + right_splits
    if aspect > 1.8:
        mid = w // 2
        return [(x, y, mid, h), (x + mid, y, w - mid, h)]
    return [(x, y, w, h)]


def extract_character(binary_row, box, padding=10):
    x, y, w, h = box
    img_h, img_w = binary_row.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    char_img = binary_row[y1:y2, x1:x2]
    if char_img.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    super_size = (IMG_SIZE * 4, IMG_SIZE * 4)
    super_res = cv2.resize(char_img, super_size, interpolation=cv2.INTER_CUBIC)
    gaussian = cv2.GaussianBlur(super_res, (0, 0), 3.0)
    sharpened = cv2.addWeighted(super_res, 2.0, gaussian, -1.0, 0)
    _, thresh = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    final = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    _, crisp = cv2.threshold(final, 127, 255, cv2.THRESH_BINARY)
    return crisp


def segment_plate(binary):
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
    
    filtered_boxes = []
    for i, box in enumerate(final_boxes):
        if i == 0:
            filtered_boxes.append(box)
        else:
            prev_box = filtered_boxes[-1]
            prev_x2 = prev_box[0] + prev_box[2]
            curr_x1 = box[0]
            gap = curr_x1 - prev_x2
            if gap >= 3:
                filtered_boxes.append(box)
    
    chars = [extract_character(row, box) for box in filtered_boxes]
    return chars, filtered_boxes, row


# ============ PREDICTION ============
def predict_char(model, classes, device, char_img):
    img = char_img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize like training
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = probs.max(1)
    top3_vals, top3_idx = torch.topk(probs, min(3, len(classes)), dim=1)
    top3 = [(classes[top3_idx[0][j].item()], top3_vals[0][j].item() * 100)
            for j in range(top3_vals.shape[1])]
    return classes[pred_idx.item()], conf.item() * 100, top3


def save_debug(row, boxes, predictions):
    scale = 4
    debug = cv2.resize(cv2.cvtColor(row, cv2.COLOR_GRAY2BGR),
                       (row.shape[1] * scale, row.shape[0] * scale))
    for (x, y, w, h), (char, conf, _) in zip(boxes, predictions):
        sx, sy, sw, sh = x*scale, y*scale, w*scale, h*scale
        color = (0, 220, 0) if conf >= 40 else (0, 140, 255)
        cv2.rectangle(debug, (sx, sy), (sx+sw, sy+sh), color, 2)
        cv2.putText(debug, f"{char} {conf:.0f}%", (sx, max(12, sy - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite("debug_result.png", debug)
    
    n = len(boxes)
    if n == 0:
        return
    cell = IMG_SIZE * 3
    sheet = np.zeros((cell + 30, cell * n, 1), dtype=np.uint8)
    for i, (box, (char, conf, _)) in enumerate(zip(boxes, predictions)):
        crop = extract_character(row, box)
        big = cv2.resize(crop, (cell, cell))
        sheet[:cell, i*cell:(i+1)*cell, 0] = big
        cv2.putText(sheet, f"{char}", (i*cell + 5, cell + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 1)
    cv2.imwrite("debug_chars.png", sheet)


def predict_plate(image_path):
    print("="*60)
    print("  PLATE RECOGNITION")
    print("="*60)
    print(f"Image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return
    
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)
    model = CharCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"✅ Model loaded | {len(classes)} classes | {device}")
    
    binary = preprocess_one_image(img)
    cv2.imwrite("temp_preprocessed.png", binary)
    
    chars, boxes, row = segment_plate(binary)
    print(f"✅ Found {len(chars)} characters")
    
    if not chars:
        print("❌ No characters detected!")
        return
    
    predictions = []
    print("\nPredictions:")
    for i, char_img in enumerate(chars):
        label, conf, top3 = predict_char(model, classes, device, char_img)
        predictions.append((label, conf, top3))
        status = "✅" if conf >= 40 else "⚠️ "
        top3_str = " | ".join([f"{c}:{v:.0f}%" for c, v in top3])
        print(f"{status} Char {i+1}: {label:2s} ({conf:5.1f}%)  [{top3_str}]")
    
    save_debug(row, boxes, predictions)
    
    plate_chars = [p[0] for p in predictions]
    plate_ltr = " ".join(plate_chars)
    avg_conf = sum(p[1] for p in predictions) / len(predictions)
    
    print(f"\n{'='*60}")
    print(f"  PLATE: {plate_ltr}")
    print(f"  Avg confidence: {avg_conf:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.11 predict.py <image_path>")
        sys.exit(1)
    predict_plate(sys.argv[1])