"""
STEP 5 - TRAIN THE CNN MODEL
==============================
Trains a neural network on your labeled characters.

Output:
  - model.pth1        → trained model
  - classes.json      → class names  
  - training_plot.png → accuracy chart

How to run:
  python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import csv
import json
from collections import Counter
import matplotlib.pyplot as plt

CHARS_FOLDER  = "characters"
LABELS_FILE   = "labels.csv"
MODEL_FILE    = "model.pth1"
CLASSES_FILE  = "classes.json"
IMG_SIZE      = 64  # Increased from 32 for better quality
BATCH_SIZE    = 64
EPOCHS        = 20
LEARNING_RATE = 0.0005
TEST_SPLIT    = 0.2


class CharDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super(CharCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),  # 64x64 input → 8x8 after 3 pooling layers
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_data():
    labels_map = {}
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == 2:
                labels_map[row[0]] = row[1]

    if not labels_map:
        raise ValueError("No labels. Run labeling.py first.")

    class_counts = Counter(labels_map.values())
    print(f"\nFound {len(labels_map)} labels, {len(class_counts)} classes")
    
    for char, count in sorted(class_counts.items(), key=lambda x: x[1]):
        print(f"  {char:3s}: {count}")

    valid_classes = {c for c, n in class_counts.items() if n >= 3}
    classes = sorted(valid_classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X, y = [], []
    skipped = 0
    
    for filename, label in labels_map.items():
        if label not in valid_classes:
            continue
        path = os.path.join(CHARS_FOLDER, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        X.append(img)
        y.append(class_to_idx[label])

    if skipped > 0:
        print(f"Skipped {skipped} missing files")

    return np.array(X), np.array(y), classes


def split_data(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_ratio))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


def run_training():
    print("="*55)
    print("  STEP 5 - CNN TRAINING")
    print("="*55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  No GPU - using CPU (slower)")

    print("\nLoading labeled data...")
    X, y, classes = load_data()
    print(f"\nTotal images: {len(X)}")
    print(f"Total classes: {len(classes)}")

    if len(X) < 50:
        print("\n❌ ERROR: Need at least 50 labeled images")
        return

    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)

    X_train, y_train, X_test, y_test = split_data(X, y, TEST_SPLIT)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(
        CharDataset(X_train, y_train, transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    test_loader = DataLoader(
        CharDataset(X_test, y_test, transform=test_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CharCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\nTraining for {EPOCHS} epochs...\n")
    train_losses, test_accs = [], []
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * test_correct / test_total
        train_losses.append(avg_loss)
        test_accs.append(test_acc)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "img_size": IMG_SIZE,
                "num_classes": len(classes)
            }, MODEL_FILE)

        marker = " ← best" if test_acc == best_acc else ""
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | "
              f"Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.1f}%{marker}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color="steelblue")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(test_accs, color="green")
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.tight_layout()
    plt.savefig("training_plot.png")

    print(f"\n{'='*55}")
    print(f"✅ Training complete!")
    print(f"Best test accuracy: {best_acc:.1f}%")
    print(f"Model saved: {MODEL_FILE}")
    print(f"{'='*55}")


if __name__ == "__main__":
    run_training()