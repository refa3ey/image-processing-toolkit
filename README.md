# 🖼️ Image Processing Toolkit

Python scripts for license plate image processing pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## 📋 Features

- **Augmentation** - Generate image variations
- **Preprocessing** - CLAHE + Adaptive Thresholding
- **Segmentation** - Character extraction with smart filtering
- **Labeling** - Manual labeling tool
- **Training** - CNN training pipeline
- **Prediction** - Inference on new images

## 🚀 Installation

```bash
git clone https://github.com/refa3ey/image-processing-toolkit.git
cd image-processing-toolkit
pip install -r requirements.txt
```

## 💻 Usage

### Augmentation
```bash
python src/augmentation.py
```

### Preprocessing
```bash
python src/preprocessing.py
```

### Segmentation
```bash
python src/segmentation.py
```

### Labeling
```bash
python src/labeling.py
```

### Training
```bash
python src/train.py
```

### Prediction
```bash
python src/predict.py samples/original/plate_001.jpg
```

## 📁 Repository Structure
## 🛠️ Technologies

- Python 3.11
- OpenCV (cv2)
- PyTorch
- NumPy
- Matplotlib

## 👤 Author

**Bilal Mahmoud Alrifaee**
- GitHub: [@refa3ey](https://github.com/refa3ey)

## 📜 License

MIT License
