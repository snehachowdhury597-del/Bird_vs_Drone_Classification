# 🐦🚁 Bird vs Drone Classification using Deep Learning

## 📌 Overview
This project focuses on building a deep learning model that can accurately distinguish between **birds and drones** from images.  
With the increasing use of drones in airspace, detecting them correctly is critical for **air safety, surveillance systems, and security applications**.

The model is trained using a convolutional neural network (CNN) to classify input images into:
- 🐦 Bird  
- 🚁 Drone  

---

## 🎯 Problem Statement
With the rise of drone usage in both civilian and military domains, differentiating drones from birds in real-time is a challenging but important task. Traditional monitoring systems often fail in:
- Low visibility conditions
- Fast-moving aerial objects
- Similar shape confusion between birds and drones

This project aims to solve that using AI-based image classification.

---

## ⚙️ Tech Stack
- Python 
- PyTorch
- OpenCV 
- NumPy
- Matplotlib
- CNN (Convolutional Neural Network)

---

## 🧠 Model Architecture

We experimented with multiple deep learning approaches to achieve optimal performance:

### 🔹 1. Custom CNN (Built from Scratch)
We designed a custom Convolutional Neural Network tailored for binary image classification.

- Multiple Convolution + ReLU layers
- MaxPooling for spatial downsampling
- Fully Connected layers for final prediction
- Sigmoid/Softmax activation for output

This model helped us understand feature extraction from scratch and baseline performance.

---

### 🔹 2. Transfer Learning (Improved Performance)
To enhance accuracy and generalization, we also used **transfer learning** with pre-trained models.

- Used models pre-trained on ImageNet (e.g., ResNet / Mobilenet / EfficientNet)
- Replaced final classification layer for 2 classes (Bird vs Drone)
- Fine-tuned top layers for domain-specific learning

This approach significantly improved:
- Training speed ⚡
- Accuracy 📈
- Generalization on unseen data 🎯

---

### 🔥 Final Approach
After comparison, transfer learning performed better and was chosen as the final model for deployment.

---

## 📊 Dataset
- Images of birds and drones collected from multiple sources
- Preprocessed with resizing, normalization, and augmentation
- Split into:
  - Training set
  - Validation set
  - Test set

---

## 🚀 Features
- Real-time image classification
- High accuracy binary classification model
- Data augmentation for better generalization
- Easy-to-use inference pipeline

---

## 🧪 Results
- Achieved high accuracy on test dataset
- Model performs well under different lighting and background conditions
- Reduced misclassification between birds and drones

*accuracy is good 95-96%*

---

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Bird_vs_Drone_Classification.git
cd Bird_vs_Drone_Classification
