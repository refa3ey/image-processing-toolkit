"""
STEP 4 - LABELING TOOL
=======================
Interactive tool to label character images.

Controls:
  Type character + Enter → save label
  S + Enter             → skip forever
  Q + Enter             → quit and save

Egyptian plate characters:
  Arabic : ا  ب  ج  د  ر  س  ص  ط  ع  ق  م  ن  ه  و  ي
  Numbers: 0  1  2  3  4  5  6  7  8  9
  
How to run:
  python labeling.py
"""

import cv2
import os
import csv
import random
from glob import glob
from collections import Counter

CHARS_FOLDER = "characters"
LABELS_FILE  = "labels.csv"
SKIPPED_FILE = "skipped.txt"
DISPLAY_SIZE = (300, 300)  # Increased from 200x200 for better visibility

# Set to [] to label ALL characters
# Or set specific chars to focus on weak classes
FOCUS_CHARS = []


def load_existing_labels():
    labeled = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) == 2:
                    labeled[row[0]] = row[1]
    return labeled


def load_skipped():
    skipped = set()
    if os.path.exists(SKIPPED_FILE):
        with open(SKIPPED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    skipped.add(name)
    return skipped


def save_skipped(filename):
    with open(SKIPPED_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def print_class_summary(labels):
    if not labels:
        return
    counts = Counter(labels.values())
    print("\nLabels per class:")
    for char, count in sorted(counts.items(), key=lambda x: x[1]):
        bar = "█" * min(count, 40)
        print(f"  {char:3s} : {count:4d}  {bar}")
    print(f"\nTotal classes: {len(counts)}")
    print(f"Total labels : {len(labels)}\n")


def run_labeling():
    all_images = sorted(glob(os.path.join(CHARS_FOLDER, "*.png")))

    if not all_images:
        print(f"ERROR: No images in '{CHARS_FOLDER}'")
        print("Run segmentation.py first.")
        return

    existing_labels = load_existing_labels()
    skipped = load_skipped()
    already_done = set(existing_labels.keys()) | skipped

    remaining = [p for p in all_images
                 if os.path.basename(p) not in already_done]

    random.seed()
    random.shuffle(remaining)

    print(f"Total characters : {len(all_images)}")
    print(f"Already labeled  : {len(existing_labels)}")
    print(f"Skipped         : {len(skipped)}")
    print(f"Remaining       : {len(remaining)}")
    
    if FOCUS_CHARS:
        print(f"\n⚠️  FOCUS MODE: Only {FOCUS_CHARS}")

    print_class_summary(existing_labels)

    print("Controls:")
    print("  Type char + Enter → label")
    print("  S + Enter         → skip")
    print("  Q + Enter         → quit\n")

    if not remaining:
        print("All done!")
        return

    with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if len(existing_labels) == 0:
            writer.writerow(["filename", "label"])

        session_count = 0

        for i, path in enumerate(remaining):
            filename = os.path.basename(path)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                save_skipped(filename)
                continue

            display = cv2.resize(img, DISPLAY_SIZE,
                                 interpolation=cv2.INTER_NEAREST)
            display_color = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

            progress = f"{i+1}/{len(remaining)}"
            cv2.putText(display_color, progress, (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

            cv2.imshow("Label this character (type in terminal)",
                       display_color)
            cv2.waitKey(1)

            label = input(f"[{i+1}/{len(remaining)}] → ").strip()

            if label.lower() == "q":
                print("Quit. Progress saved.")
                break
            elif label.lower() == "s" or label == "":
                save_skipped(filename)
                continue
            else:
                if FOCUS_CHARS and label not in FOCUS_CHARS:
                    print(f"  Note: {label} not in focus list")

                writer.writerow([filename, label])
                f.flush()
                session_count += 1

                if session_count % 100 == 0:
                    updated = load_existing_labels()
                    print(f"\n--- Labeled {session_count} this session ---")
                    print_class_summary(updated)

    cv2.destroyAllWindows()

    final = load_existing_labels()
    print(f"\n✅ Session complete!")
    print(f"Labeled: {session_count}")
    print_class_summary(final)


if __name__ == "__main__":
    run_labeling()