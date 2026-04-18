[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./images/sample_cnn.png "Sample CNN"
[image5]: ./images/sample_human_output.png "Sample Human Output"

# 🐕 Dog Breed Classification — CNN from Scratch + ResNet-50 Transfer Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/Udacity-AI_Nanodegree-02B3E4)

## 🚀 Overview

An image classification pipeline that accepts any user-supplied image and determines: (1) if it's a dog, it identifies the breed from 133 classes, or (2) if it's a human, it identifies the most resembling dog breed. The project builds two CNNs — one from scratch (35% accuracy) and one using ResNet-50 transfer learning (79% accuracy) — combined with OpenCV Haar cascades for face detection and VGG-16 for dog detection.

![Sample Output][image1]

## 📊 Results

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Random Baseline | 0.75% (1/133) | — |
| CNN from Scratch (5 conv layers, 40 epochs) | **35%** | 2.779 |
| ResNet-50 Transfer Learning | **79%** | 0.901 |

**Detection Accuracy:**
- Human face detector (Haar cascade): 98% on human images, 83% rejection on dog images
- Dog detector (VGG-16, ImageNet classes 151–268): used for dog/not-dog classification

## ✨ Key Features

- **Full Detection Pipeline** — Input any image → detect human (Haar cascade) or dog (VGG-16) → classify breed (ResNet-50) → output breed name or resembling breed
- **CNN from Scratch** — 5 convolutional layers (3→16→32→64→128→256) with max pooling, dropout, and a single FC layer mapping to 133 breeds
- **Transfer Learning** — Pre-trained ResNet-50 with frozen feature layers and replaced final FC layer (`fc: 2048 → 133`), achieving 79% accuracy
- **Human Face Detection** — OpenCV Haar cascade (`haarcascade_frontalface_alt.xml`) for detecting human faces in images
- **Dog Detection** — VGG-16 pre-trained on ImageNet; classes 151–268 correspond to dog breeds, used as a binary dog detector

![Sample CNN][image4]

## 🧠 Technical Highlights

**CNN from Scratch:**
```
Input (3, 224, 224)
  → Conv2d(3→16, 3×3) → ReLU → MaxPool(2×2)     → (16, 112, 112)
  → Conv2d(16→32, 3×3) → ReLU → MaxPool(2×2)     → (32, 56, 56)
  → Conv2d(32→64, 3×3) → ReLU → MaxPool(2×2)     → (64, 28, 28)
  → Conv2d(64→128, 3×3) → ReLU → MaxPool(2×2)    → (128, 14, 14)
  → Conv2d(128→256, 3×3) → ReLU → MaxPool(2×2)   → (256, 7, 7)
  → Flatten → Dropout(0.2) → Linear(256*7*7 → 133)
```

**Transfer Learning (ResNet-50):**
- All convolutional layers frozen (pre-trained on ImageNet)
- Final `fc` layer replaced: `Linear(2048 → 133)` for 133 dog breeds
- Only the new FC layer is trained, leveraging ResNet-50's learned feature representations

**Why This Problem Is Hard:**
- Minimal inter-class variation: Brittany vs. Welsh Springer Spaniel look nearly identical
- High intra-class variation: Labradors come in yellow, chocolate, and black
- 133 classes with limited training data per breed

![Sample Human Output][image5]

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch |
| Transfer Learning | ResNet-50 (ImageNet pre-trained) |
| Dog Detection | VGG-16 (ImageNet classes 151–268) |
| Face Detection | OpenCV Haar Cascade |
| Loss | CrossEntropyLoss |
| Optimizer | SGD (lr=0.01) for scratch, Adam for transfer |
| Dataset | 8,351 dog images across 133 breeds |

## 🏗 Pipeline Architecture

```
User-Supplied Image
    │
    ├── OpenCV Haar Cascade → Human detected?
    │   └── Yes → ResNet-50 → "You look like a {breed}!"
    │
    ├── VGG-16 (ImageNet 151-268) → Dog detected?
    │   └── Yes → ResNet-50 → "Predicted breed: {breed}"
    │
    └── Neither → "Error: No human or dog detected"
```

## ⚡ Getting Started

```bash
git clone https://github.com/jashjain21/CNN-Dog_breed_classification.git
cd CNN-Dog_breed_classification

pip install torch torchvision opencv-python numpy matplotlib

# Download datasets (required)
# Dog images: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
# Human faces: http://vis-www.cs.umass.edu/lfw/lfw.tgz

jupyter notebook dog_app.ipynb
```

Test images are provided in `upload_images_for_testing/` for quick evaluation.

## 🔍 What This Project Demonstrates

- **Transfer Learning** — Replacing the final layer of a pre-trained ResNet-50 and fine-tuning for a domain-specific task, jumping from 35% (scratch) to 79% accuracy
- **Multi-Model Pipeline** — Chaining Haar cascades, VGG-16, and ResNet-50 into a single inference pipeline that handles different input types (human, dog, other)
- **CNN Architecture Design** — Building a 5-layer CNN from scratch with progressive channel depth (16→32→64→128→256) and evaluating its limitations
- **Computer Vision Fundamentals** — Face detection, image classification, data augmentation, and the challenges of fine-grained visual categorization (133 visually similar breeds)

## 🚧 Limitations / Future Improvements

- **79% Accuracy** — While above the 60% threshold, fine-tuning more ResNet layers (not just the FC) or using a larger model (ResNet-152, EfficientNet) could push accuracy higher
- **No Data Augmentation on Transfer Model** — Adding random rotations, flips, and color jitter during training would improve generalization
- **Haar Cascade Limitations** — The face detector misclassifies 17% of dog images as containing faces; a deep learning face detector (MTCNN, RetinaFace) would be more robust
- **No Top-K Predictions** — Only the top-1 breed is returned; showing top-3 with confidence scores would be more useful for similar-looking breeds
- **No Web/Mobile Interface** — The pipeline runs in a notebook; wrapping it in a Flask/FastAPI app would make it user-facing

## Author
Jash Jain : [LinkedIn](https://www.linkedin.com/in/jash-jain-bb659a132)
